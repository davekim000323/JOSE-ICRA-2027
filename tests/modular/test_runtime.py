import math

import torch

from modular.embodiment import load_embodiment
from modular.features import FeatureBuilder
from modular.runtime import JoseRuntime
from modular.signals import JointReading

from helpers import ConstantModel

BOX_NAMES = (
    "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",
    "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
    "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
    "left_hand_contact_force", "right_hand_contact_force", "payload_mass",
)


def make(spec_name="g1_29dof", outputs=None, window=4):
    spec = load_embodiment(spec_name)
    fb = FeatureBuilder(spec, "native")
    outputs = outputs if outputs is not None else torch.tensor([[0.1, 0, 0, 0, 0, 0.5, 0, 0, -2.0, 0.3, 0.4, 1.5]])
    model = ConstantModel(outputs)
    return spec, JoseRuntime(model, fb, BOX_NAMES, window), model


def test_parses_all_fields_and_normalizes_gravity():
    spec, rt, _ = make()
    est = rt.step(JointReading(torch.zeros(spec.num_joints), torch.zeros(spec.num_joints)))
    assert torch.allclose(est.projected_gravity, torch.tensor([[0.0, 0.0, -1.0]]))
    assert set(est.contact) == {"left_hand", "right_hand"}
    assert torch.allclose(est.payload, torch.tensor([1.5]))
    assert est.base_lin_vel is not None


def test_history_rolls_and_resets_to_zero():
    spec, rt, model = make(window=3)
    for k in range(1, 5):
        rt.step(JointReading(torch.full((spec.num_joints,), float(k)), torch.zeros(spec.num_joints)))
    last = model.inputs[-1][0, :, 0]
    assert torch.equal(last, torch.tensor([2.0, 3.0, 4.0]))
    rt.reset()
    rt.step(JointReading(torch.full((spec.num_joints,), 9.0), torch.zeros(spec.num_joints)))
    assert torch.equal(model.inputs[-1][0, :, 0], torch.tensor([0.0, 0.0, 9.0]))


def test_yaw_integrates_estimated_rate():
    spec, rt, _ = make()
    for _ in range(50):
        est = rt.step(JointReading(torch.zeros(spec.num_joints), torch.zeros(spec.num_joints)))
    assert math.isclose(float(est.yaw), 50 * spec.control_dt * 0.5, rel_tol=1e-5)


def test_missing_gravity_output_is_rejected():
    spec = load_embodiment("g1_29dof")
    try:
        JoseRuntime(ConstantModel(torch.zeros(1, 3)), FeatureBuilder(spec), ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"], 2)
    except ValueError as err:
        assert "projected_gravity" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_loads_a_pipeline_checkpoint_without_isaac(tmp_path, schema):
    """Write a checkpoint in save_jose_checkpoint's format, then load it on CPU."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec_m = importlib.util.spec_from_file_location("_m", root / "estimator" / "models.py")
    models = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(models)
    torch.manual_seed(0)
    net = models.build_estimator("LSTM", 87, 9, 32, 1)
    payload = {
        "jose_schema_version": schema.SCHEMA_VERSION, "kind": "estimator", "adapter": "ppo_walk",
        "observation_schema": schema.PPO_WALK_OBSERVATION_SCHEMA.to_dict(), "joint_preset": "all",
        "imu_input": False, "torque_input": True, "torque_source": "pd", "action_input": False,
        "velocity_source": "sim_joint_velocity", "window": 25, "model_config": net.config(),
        "model_state_dict": net.state_dict(),
    }
    path = tmp_path / "jose.pt"
    torch.save(payload, path)
    spec = load_embodiment("g1_29dof")
    rt = JoseRuntime.from_checkpoint(path, FeatureBuilder(spec, "native", use_torque=True))
    assert rt.window == 25
    reading = JointReading(torch.randn(29), torch.randn(29), torch.randn(29))
    est = rt.step(reading)
    history = torch.zeros(1, 25, 87)
    history[:, -1] = FeatureBuilder(spec, "native", True)(reading)
    expected = net.predict(history)
    assert torch.allclose(est.raw, expected, atol=1e-6)
