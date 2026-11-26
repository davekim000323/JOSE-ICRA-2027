"""End-to-end wiring. One shared estimator, many robots, many VLAs, no Isaac Lab."""
import importlib.util
from pathlib import Path

import pytest
import torch

from modular import FeatureBuilder, JointReading, JoseRuntime, build_stack, load_embodiment
from modular.features import CANONICAL_SLOTS

ROOT = Path(__file__).resolve().parents[2]
NAMES = [f"{f}_{a}" for f in ("base_lin_vel", "base_ang_vel", "projected_gravity") for a in "xyz"]
ROBOT_PORTS = {"g1_29dof": "unitree_rl_gym", "g1_23dof": "unitree_rl_gym", "h1": "unitree_rl_gym",
               "h1_2": "unitree_rl_gym", "t1": "booster_t1"}
VLAS = ["groot_dict", "openhlm_body", "openpi_flat", "openvla_tokens", "psi0_gripper", "sonic_latent", "wholebodyvla_flags"]


@pytest.fixture(scope="module")
def shared_model():
    spec = importlib.util.spec_from_file_location("_m", ROOT / "estimator" / "models.py")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    torch.manual_seed(0)
    net = models.build_estimator("LSTM", 4 * len(CANONICAL_SLOTS), 9, 32, 1).eval()
    return net


def reading(spec, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = spec.num_joints
    return JointReading(torch.randn(n, generator=g) * 0.2, torch.randn(n, generator=g), torch.randn(n, generator=g) * 5)


@pytest.mark.parametrize("robot", sorted(ROBOT_PORTS))
def test_one_canonical_estimator_drives_every_robot(robot, shared_model):
    def factory(spec):
        return JoseRuntime(shared_model.predict, FeatureBuilder(spec, "canonical", use_torque=True), NAMES, window=25)

    for port in (ROBOT_PORTS[robot], "imu_message"):
        stack = build_stack(robot, port, factory)
        for t in range(5):
            r = reading(stack.spec, t)
            if port == "imu_message":
                msg = {"imu_state": {"quaternion": [0.0] * 4, "gyroscope": [0.0] * 3, "rpy": [0.0] * 3, "accelerometer": [0.0] * 3}}
                out = stack.tick(r, msg)
                assert abs(sum(v * v for v in out["imu_state"]["quaternion"]) - 1.0) < 1e-5
            else:
                obs = stack.tick(r, torch.zeros(47))
                assert torch.isfinite(obs).all() and obs[:6].abs().sum() > 0
        assert torch.isclose(stack.last_estimate.projected_gravity.norm(), torch.tensor(1.0))


@pytest.mark.parametrize("vla", VLAS)
def test_every_vla_preset_runs_on_the_same_g1_stack(vla, shared_model):
    def factory(spec):
        return JoseRuntime(shared_model.predict, FeatureBuilder(spec, "canonical", use_torque=True), NAMES, window=25)

    stack = build_stack("g1_29dof_gripper", "imu_message", factory, high_level=vla)
    r = reading(stack.spec)
    stack.tick(r, {"imu_state": {"quaternion": [0.0] * 4, "gyroscope": [0.0] * 3, "rpy": [0.0] * 3, "accelerometer": [0.0] * 3}})
    state = stack.high_level_state(r)
    hl = stack.high_level
    if hl.action_format == "dict":
        raw = {f["key"]: torch.rand(hl.horizon, hl._dims(f)) for f in hl.action_fields if f["kind"] != "skip"}
    elif hl.action_format == "tokens":
        raw = torch.randint(0, hl.bins, (hl.action_dim,))
    else:
        width = hl.action_pad or hl.action_dim
        raw = torch.rand(hl.horizon, width) if hl.horizon > 1 else torch.rand(width)
    cmds = stack.command(raw, r)
    assert len(cmds) == min(hl.horizon, hl.execute)
    assert state is None or isinstance(state, (dict, torch.Tensor))
