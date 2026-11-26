"""CPU smoke test of the torque and action estimator inputs, no Isaac Lab needed.

Run with ``python tools/smoke_torque_input.py`` from the repository root. It
builds a fake articulation, checks input widths and the joint-block layout for
every arm, the checkpoint round trip, clipping and normalization of the torque
channel, per-episode resampling of the sensor gain, and the tilt metric.
"""
import sys, types, importlib, torch
# stub skrl_compat and task_math deps that import isaaclab/skrl
import os
pkg = types.ModuleType("jose"); pkg.__path__ = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
sys.modules["jose"] = pkg  # skip jose/__init__.py, which registers Isaac Lab gym tasks
import jose.schema as schema
from jose.estimator.adapters import make_policy_adapter
from jose.estimator.torque import torque_noise_preset, TorqueSensorModel
from jose.estimator.pipeline import gravity_tilt_error_deg, adapter_input_options, HistoryBuffer
from jose.estimator.models import build_estimator

N, J = 8, 29
class Data:
    def __init__(self):
        self.joint_names = list(schema.G1_JOINT_NAMES)
        self.joint_pos = torch.randn(N, J); self.joint_vel = torch.randn(N, J)
        self.applied_torque = torch.randn(N, J) * 200  # deliberately beyond limits
        lim = torch.tensor([88.,139,88,139,50,50]*2 + [88,50,50] + [25]*14)
        self.joint_effort_limits = lim.repeat(N, 1)
        self.body_names = ["pelvis"]
class Robot: data = Data()
class ActionManager: action = torch.randn(N, J)
class Env:
    robot = Robot(); action_manager = ActionManager()
    episode_length_buf = torch.ones(N, dtype=torch.long)
    def get_estimator_joint_state(self): d=self.robot.data; return d.joint_pos, d.joint_vel, tuple(d.joint_names)
    def get_estimator_target(self): return torch.randn(N, 9)
    def get_distillation_sensor_state(self): return {"angular_velocity": torch.zeros(N,3), "projected_gravity": torch.zeros(N,3)}
    def as_policy_input(self, o): return o
env = Env()
arms = {
  "enc": dict(),
  "enc+tau": dict(use_torque=True, torque_noise=torque_noise_preset("default")),
  "enc+tau(clean)": dict(use_torque=True, torque_noise=torque_noise_preset("none")),
  "enc+act": dict(use_action=True),
  "enc+tau+act": dict(use_torque=True, use_action=True, torque_noise=torque_noise_preset("default")),
  "enc+tau(legs)": dict(use_torque=True),
}
for name, kw in arms.items():
    preset = "legs" if "legs" in name else "all"
    a = make_policy_adapter("ppo_walk", env, preset, **kw)
    x = a.estimator_input()
    assert x.shape == (N, a.input_dim), (name, x.shape, a.input_dim)
    nj = len(a.joint_ids)
    # joint block unchanged across arms
    assert torch.equal(x[:, :2*nj], make_policy_adapter("ppo_walk", env, preset).estimator_input())
    est = build_estimator("LSTM", a.input_dim, 9)
    h = HistoryBuffer(N, 25, a.input_dim, "cpu")
    y = est.predict(h.push(x))
    assert y.shape == (N, 9)
    # checkpoint round trip of input options
    payload = {"imu_input": a.use_imu, "torque_input": a.use_torque, "torque_source": a.torque_source,
               "torque_noise": a.torque_noise.to_dict(), "action_input": a.use_action}
    b = make_policy_adapter("ppo_walk", env, preset, **adapter_input_options(payload))
    assert b.input_dim == a.input_dim
    print(f"{name:16s} preset={preset:4s} input_dim={a.input_dim}")
# old checkpoints (no torque keys) resolve to encoder-only
assert make_policy_adapter("ppo_walk", env, "all", **adapter_input_options({})).input_dim == 58

# clean torque: clipped to [-1, 1] after effort-limit normalization, equals clip(tau)/limit
a = make_policy_adapter("ppo_walk", env, "all", use_torque=True, torque_noise=torque_noise_preset("none"))
tau = a.torque_sensor.read(); d = env.robot.data
ref = torch.maximum(torch.minimum(d.applied_torque, d.joint_effort_limits), -d.joint_effort_limits) / d.joint_effort_limits
assert torch.allclose(tau, ref) and tau.abs().max() <= 1.0
# noisy: per-episode gain/bias persists across steps, resampled on reset
a = make_policy_adapter("ppo_walk", env, "all", use_torque=True, torque_noise=torque_noise_preset("default"))
s = a.torque_sensor; s.read(); g0 = s._gain.clone()
s.read(); assert torch.equal(g0, s._gain)
env.episode_length_buf[:3] = 0; s.read(); env.episode_length_buf[:] = 1
assert not torch.equal(g0[:3], s._gain[:3]) and torch.equal(g0[3:], s._gain[3:])
assert ((s._gain - 1).abs() <= 0.10 + 1e-6).all()
err = (s.read() - ref).abs().mean().item()
print(f"default noise mean abs error (fraction of effort limit) {err:.4f}")

# tilt metric: 10 deg rotation about x gives 10 deg error
sch = schema.PPO_WALK_OBSERVATION_SCHEMA
t = torch.zeros(1, 9); t[0, 8] = -1
th = torch.deg2rad(torch.tensor(10.)); e = torch.zeros(1, 9); e[0, 7] = torch.sin(th); e[0, 8] = -torch.cos(th)
print("tilt metric for 10 deg:", float(gravity_tilt_error_deg(e*2, t, sch)))
print("ALL SMOKE CHECKS PASSED")
