import pytest
import torch

from modular.embodiment import EmbodimentSpec, list_embodiments, load_embodiment

EXPECTED = {"g1_29dof": 29, "g1_23dof": 23, "g1_29dof_gripper": 31, "h1": 19, "h1_2": 27, "t1": 23}


def test_every_shipped_embodiment_loads():
    assert set(EXPECTED) <= set(list_embodiments())
    for name, n in EXPECTED.items():
        spec = load_embodiment(name)
        assert spec.num_joints == n
        assert len(spec.effort_limit) == n and min(spec.effort_limit) > 0
        assert len(spec.default_q) == n


def test_g1_joint_order_is_the_training_order(schema):
    assert load_embodiment("g1_29dof").joint_names == list(schema.G1_JOINT_NAMES)


def test_imu_frames():
    assert load_embodiment("g1_29dof").imu.is_identity
    assert load_embodiment("t1").imu.is_identity
    for name in ("h1", "h1_2"):
        spec = load_embodiment(name)
        assert spec.imu.joints == ["torso_joint"] and spec.imu.frame == "torso"


def test_gripper_spec_and_estimator_excludes_grippers():
    spec = load_embodiment("g1_29dof_gripper")
    assert len(spec.estimator_joints) == 29
    assert "left_gripper_joint" not in spec.estimator_joints
    g = spec.gripper("left")
    closure = torch.tensor([0.0, 0.5, 1.0, 2.0])
    assert torch.allclose(g.to_unit(g.from_unit(closure)), closure.clamp(0, 1))


def test_current_source_converts_with_torque_constant():
    data = {"name": "toy", "joint_names": ["a", "b"], "torque": {"source": "current", "constant": [2.0, 0.5]}}
    spec = EmbodimentSpec.from_dict(data)
    assert torch.allclose(spec.joint_torque(torch.tensor([[1.0, 4.0]])), torch.tensor([[2.0, 2.0]]))


@pytest.mark.parametrize("bad", [
    {"estimator_joints": ["missing"]},
    {"imu": {"chain": [{"joint": "missing", "axis": [0, 0, 1]}]}},
    {"torque": {"source": "current"}},
    {"joint_names": ["a", "a"]},
])
def test_invalid_specs_are_rejected(bad):
    data = {"name": "toy", "joint_names": ["a", "b"], **bad}
    with pytest.raises((ValueError, KeyError)):
        EmbodimentSpec.from_dict(data)
