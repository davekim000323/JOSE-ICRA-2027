"""CPU smoke test of the box-carry task math, no Isaac Lab needed.

Run with ``python tools/smoke_box_carry.py`` from the repository root. It stubs
the few Isaac Lab symbols ``ppo_walk/mdp/box_carry_mdp.py`` imports, builds a
fake scene, and checks grasp geometry, the reward gates, the scene reset, the
curriculum, the 540-D observation schema and the JOSE estimate injection.
"""
import math
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- Isaac Lab stubs
def quat_apply(q, v):
    w, x, y, z = q.unbind(-1)
    qv = torch.stack((x, y, z), -1)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)


def quat_from_euler_xyz(r, p, y):
    cr, sr = torch.cos(r / 2), torch.sin(r / 2)
    cp, sp = torch.cos(p / 2), torch.sin(p / 2)
    cy, sy = torch.cos(y / 2), torch.sin(y / 2)
    return torch.stack((
        cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


class SceneEntityCfg:
    def __init__(self, name, body_names=None, joint_names=None):
        self.name, self.body_ids = name, [0, 1]


for name in ("isaaclab", "isaaclab.utils", "isaaclab.utils.math", "isaaclab.managers", "isaaclab.envs"):
    sys.modules[name] = types.ModuleType(name)
m = sys.modules["isaaclab.utils.math"]
m.quat_apply, m.quat_apply_inverse, m.quat_from_euler_xyz, m.yaw_quat = quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat
sys.modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg

pkg = types.ModuleType("jose"); pkg.__path__ = [ROOT]; sys.modules["jose"] = pkg
for sub in ("ppo_walk", "ppo_walk.mdp"):
    mod = types.ModuleType(f"jose.{sub}"); mod.__path__ = [os.path.join(ROOT, *sub.split("."))]; sys.modules[f"jose.{sub}"] = mod

import jose.ppo_walk.mdp.box_carry_mdp as box  # noqa: E402
import jose.schema as schema  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.torque import torque_noise_preset  # noqa: E402

N = 6
ID = torch.tensor([1.0, 0, 0, 0])


# ---------------------------------------------------------------- fake scene
class D:
    pass


class Rigid:
    def __init__(self, pos, quat=None):
        self.data = D()
        self.data.root_pos_w = pos.clone()
        self.data.root_quat_w = (quat if quat is not None else ID.repeat(N, 1)).clone()
        self.data.root_lin_vel_w = torch.zeros(N, 3)
        self.root_physx_view = D()
        self.root_physx_view.get_masses = lambda: torch.linspace(0.5, 3.0, N).unsqueeze(-1)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:]

    def write_root_velocity_to_sim(self, vel, env_ids):
        self.data.root_lin_vel_w[env_ids] = vel[:, :3]


class Robot(Rigid):
    BODIES = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]

    def __init__(self):
        super().__init__(torch.tensor([0.0, 0.0, 0.78]).repeat(N, 1))
        self.data.body_pos_w = torch.zeros(N, 3, 3)
        self.data.body_quat_w = ID.repeat(N, 3, 1)
        self.data.joint_names = list(schema.G1_JOINT_NAMES)
        self.data.joint_pos = torch.zeros(N, 29); self.data.joint_vel = torch.zeros(N, 29)
        self.data.applied_torque = torch.zeros(N, 29)
        self.data.joint_effort_limits = torch.full((N, 29), 25.0)
        self.data.body_names = self.BODIES

    def find_bodies(self, name):
        return [self.BODIES.index(name)], [name]

    def put_palms(self, points):
        # palm = wrist + HAND_OFFSET along wrist x, wrist quat identity
        self.data.body_pos_w[:, 1:3] = points - torch.tensor([box.HAND_OFFSET, 0, 0])


class Contact:
    def __init__(self):
        self.data = D(); self.data.force_matrix_w = torch.zeros(N, 3, 1, 3)


class FeetSensor:
    def __init__(self):
        self.data = D()
        self.data.current_air_time = torch.tensor([[0.3, 0.0]]).repeat(N, 1)
        self.data.current_contact_time = torch.tensor([[0.0, 0.3]]).repeat(N, 1)


class Scene(dict):
    pass


class TM:
    def __init__(self):
        self.terminated = torch.zeros(N, dtype=torch.bool)


class Env:
    num_envs = N; device = "cpu"; common_step_counter = 1
    episode_length_buf = torch.ones(N, dtype=torch.long)

    def __init__(self):
        self.scene = Scene(robot=Robot(), box=Rigid(torch.tensor([0.45, 0.0, box.BOX_SPAWN_Z]).repeat(N, 1)),
                           pedestal=Rigid(torch.zeros(N, 3)))
        self.scene.sensors = {"left_hand_contact": Contact(), "right_hand_contact": Contact(),
                              "contact_forces": FeetSensor()}
        self.termination_manager = TM()
        self.action_manager = D(); self.action_manager.action = torch.zeros(N, 29)
        self.robot = self.scene["robot"]

    def get_estimator_joint_state(self):
        d = self.robot.data; return d.joint_pos, d.joint_vel, tuple(d.joint_names)

    def get_estimator_target(self):
        return torch.randn(N, 12)

    def as_policy_input(self, o):
        return o


def close(a, b, tol=1e-4):
    return torch.allclose(a, b if torch.is_tensor(b) else torch.full_like(a, b), atol=tol)


env = Env()
robot, bx = env.scene["robot"], env.scene["box"]
L, R = env.scene.sensors["left_hand_contact"], env.scene.sensors["right_hand_contact"]

# 1. grasp points lie on the faces pointing along the robot's left and right, for any box yaw
for yaw in (0.0, 0.3, math.pi / 2 + 0.2, math.pi, -2.5):
    bx.data.root_quat_w = quat_from_euler_xyz(torch.zeros(N), torch.zeros(N), torch.full((N,), yaw))
    g = box.grasp_points_w(env)
    assert (g[:, 0, 1] > 0.09).all() and (g[:, 1, 1] < -0.09).all(), (yaw, g[0])
    assert close((g[:, 0] - g[:, 1]).norm(dim=-1), box.BOX_SIZE[0] + 0.02)
bx.data.root_quat_w = ID.repeat(N, 1)
g = box.grasp_points_w(env)
assert close(g[:, 0], torch.tensor([0.45, 0.15, box.BOX_SPAWN_Z]))
print("grasp geometry ok")

# 2. reach is 1 with palms on the grasp points, small with arms at rest
robot.put_palms(g)
assert close(box.reach_grasp_points(env), 1.0)
robot.put_palms(torch.tensor([[0.2, 0.2, 0.9], [0.2, -0.2, 0.9]]).expand(N, 2, 3))
assert (box.reach_grasp_points(env) < 0.2).all()
print("reach ok")

# 3. gates. One hand pressing pays nothing, both pay, lift needs both and height
L.data.force_matrix_w[:, 2, 0] = torch.tensor([0, -40.0, 0])
assert close(box.squeeze(env), 0.0) and close(box.lift_box(env), 0.0)
R.data.force_matrix_w[:, 2, 0] = torch.tensor([0, 20.0, 0])
assert close(box.squeeze(env), 20 / 30)
assert close(box.hand_contact_force_obs(env), torch.tensor([0.8, 0.4]).expand(N, 2))
assert close(box.lift_box(env), 0.0) and close(box.carry_to_goal(env), 0.0)
assert close(box.approach_box(env), math.exp(-(0.10 / 0.6) ** 2))  # box 0.45 m away, 0.10 m beyond standoff, gate open
bx.data.root_pos_w[:, 2] = box.BOX_SPAWN_Z + box.LIFT_HEIGHT
assert close(box.lift_box(env), 1.0)
assert close(box.approach_box(env), 0.0), "approach must switch off once lifted"
box.goal_pos_w(env)[:] = bx.data.root_pos_w
assert close(box.carry_to_goal(env), 1.0) and close(box.hold_at_goal(env), 1.0) and close(box.delivered(env), 1.0)
L.data.force_matrix_w.zero_()
assert close(box.delivered(env), 0.0), "delivery needs the grasp"
print("reward gates ok")

# 4. stepping reward only when far from the target
assert close(box.feet_air_time_when_far(env, SceneEntityCfg("contact_forces")), 0.0)
bx.data.root_pos_w[:, :2] = torch.tensor([2.0, 0.0])
assert close(box.feet_air_time_when_far(env, SceneEntityCfg("contact_forces")), 0.3)
print("feet air time gate ok")

# 5. heading reward
bx.data.root_pos_w[:, :2] = torch.tensor([2.0, 0.0])
assert close(box.face_target(env), 1.0)
bx.data.root_pos_w[:, :2] = torch.tensor([-2.0, 0.0])
assert close(box.face_target(env), 0.0)
print("heading ok")

# 6. reset places the box on the pedestal in reach at level 0, goal clear and above spawn
torch.manual_seed(0)
for level in (0, 9):
    box.curriculum_level(env)[:] = level
    for _ in range(50):
        box.reset_box_scene(env, torch.arange(N))
        d = bx.data.root_pos_w[:, :2].norm(dim=-1)
        lim = box.level_ranges(torch.tensor(level))
        assert ((d >= 0.399) & (d <= float(lim["box_dist_max"]) + 1e-4)).all()
        bearing = torch.atan2(bx.data.root_pos_w[:, 1], bx.data.root_pos_w[:, 0]).abs()
        assert (bearing <= float(lim["box_bearing_max"]) + 1e-4).all()
        assert close(bx.data.root_pos_w[:, 2], box.BOX_SPAWN_Z)
        assert close(env.scene["pedestal"].data.root_pos_w[:, :2], bx.data.root_pos_w[:, :2])
        gd = (box.goal_pos_w(env)[:, :2] - bx.data.root_pos_w[:, :2]).norm(dim=-1)
        assert (gd >= 0.299).all() and (gd <= float(lim["goal_dist_max"]) + 1e-4).all()
print("reset ok")

# 7. curriculum: success up, failure down, clamp
box.curriculum_level(env)[:] = 3
box.reset_box_scene(env, torch.arange(N))
bx.data.root_pos_w[:] = box.goal_pos_w(env)
robot.put_palms(box.grasp_points_w(env))
L.data.force_matrix_w[:, 2, 0] = torch.tensor([0, -40.0, 0])
R.data.force_matrix_w[:, 2, 0] = torch.tensor([0, 40.0, 0])
assert close(box.delivered(env), 1.0)
box.box_level_curriculum(env, torch.tensor([0, 1]))
L.data.force_matrix_w.zero_()
env.termination_manager.terminated[2] = True
box.box_level_curriculum(env, torch.tensor([2, 3]))
lv = box.curriculum_level(env)
assert lv.tolist()[:4] == [4, 4, 2, 3], lv
print("curriculum ok")

# 8. termination and payload
bx.data.root_pos_w[:, 2] = 0.2
assert box.box_dropped(env).all()
assert close(box.payload_mass_obs(env).squeeze(-1), torch.linspace(0.5, 3.0, N))
print("termination and payload ok")

# 9. schema and JOSE injection over the full history of all five estimated terms
sch = schema.BOX_CARRY_OBSERVATION_SCHEMA
assert sch.policy_dim == 540 and sch.estimator_target_dim == 12
hist = schema.box_carry_history_target_indices()
assert len(hist) == 60 and len(set(hist)) == 60
assert sch.joint_position_start == 12 * 5 + 9 * 5 + 4 * 29  # newest joint_pos frame
for arm in (dict(), dict(use_torque=True, torque_noise=torque_noise_preset("default"))):
    ad = make_policy_adapter("box_carry", env, **arm)
    assert ad.name() == "box_carry" and ad.input_dim == 58 + (29 if arm else 0)
    assert ad.estimator_input().shape == (N, ad.input_dim)
    obs = torch.zeros(N, 540)
    est = torch.arange(12, dtype=torch.float32).repeat(N, 1)
    out = ad.inject_estimate(obs, est)
    ad._ring = None
    newest = out[:, list(sch.estimator_target_indices)]
    expected = est * torch.tensor(schema.box_carry_target_scales())
    assert close(newest, expected), newest[0]
    untouched = [i for i in range(540) if i not in set(hist)]
    assert close(out[:, untouched], 0.0)
print("schema and injection ok")
print("ALL BOX-CARRY SMOKE CHECKS PASSED")
