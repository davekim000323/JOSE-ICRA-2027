import torch

from modular.embodiment import load_embodiment
from modular.features import CANONICAL_SLOTS, FeatureBuilder
from modular.signals import JointReading


def reading(spec, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = spec.num_joints
    return JointReading(torch.randn((batch, n), generator=g), torch.randn((batch, n), generator=g),
                        torch.randn((batch, n), generator=g) * 50)


def test_native_g1_matches_training_adapter_layout():
    """q block, dq block, torque/limit block, which is estimator/adapters.py with use_torque."""
    spec = load_embodiment("g1_29dof")
    r = reading(spec)
    x = FeatureBuilder(spec, "native", use_torque=True)(r)
    assert x.shape == (2, 87)
    limit = torch.tensor(spec.effort_limit)
    assert torch.equal(x[:, :29], r.q) and torch.equal(x[:, 29:58], r.dq)
    assert torch.allclose(x[:, 58:], torch.maximum(torch.minimum(r.effort, limit), -limit) / limit)
    assert FeatureBuilder(spec, "native").input_dim == 58


def test_gripper_variant_feeds_the_same_29_joints():
    base, grip = load_embodiment("g1_29dof"), load_embodiment("g1_29dof_gripper")
    r = reading(grip)
    r_body = JointReading(r.q[:, :29], r.dq[:, :29], r.effort[:, :29])
    assert torch.equal(FeatureBuilder(grip, "native", True)(r), FeatureBuilder(base, "native", True)(r_body))


def test_canonical_width_is_robot_independent():
    widths = set()
    for name in ("g1_29dof", "g1_23dof", "h1", "h1_2", "t1"):
        spec = load_embodiment(name)
        fb = FeatureBuilder(spec, "canonical", use_torque=True)
        x = fb(reading(spec))
        widths.add(x.shape[-1])
        mask = x[:, -len(CANONICAL_SLOTS):]
        assert int(mask[0].sum()) == spec.num_joints - len(fb.unmapped_joints)
        assert fb.unmapped_joints == []
    assert widths == {4 * len(CANONICAL_SLOTS)}


def test_canonical_puts_h1_torso_in_waist_yaw():
    spec = load_embodiment("h1")
    r = reading(spec)
    x = FeatureBuilder(spec, "canonical")(r)
    n = len(CANONICAL_SLOTS)
    slot = CANONICAL_SLOTS.index("waist_yaw")
    j = spec.joint_names.index("torso_joint")
    assert torch.allclose(x[:, slot], r.q[:, j] - spec.default_q[j])
    assert torch.allclose(x[:, n + slot], r.dq[:, j])
