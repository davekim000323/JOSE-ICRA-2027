import pytest
import torch

from modular.embodiment import load_embodiment
from modular.highlevel import list_high_level_presets, make_high_level_adapter
from modular.signals import JointReading, StateEstimate

SPEC = load_embodiment("g1_29dof_gripper")
PRESETS = ["groot_dict", "openhlm_body", "openpi_flat", "openvla_tokens", "psi0_gripper", "sonic_latent", "wholebodyvla_flags"]


def reading(batch=1):
    g = torch.Generator().manual_seed(0)
    return JointReading(torch.rand((batch, SPEC.num_joints), generator=g), torch.randn((batch, SPEC.num_joints), generator=g),
                        torch.randn((batch, SPEC.num_joints), generator=g))


def est(batch=1):
    return StateEstimate(torch.tensor([[0.1, -0.2, -0.97]]).expand(batch, 3), torch.tensor([[0.0, 0.0, 0.3]]).expand(batch, 3),
                         yaw=torch.zeros(batch))


def test_all_presets_load_on_g1_with_grippers():
    assert set(PRESETS) <= set(list_high_level_presets())
    for name in PRESETS:
        make_high_level_adapter(SPEC, name)


def test_openpi_state_padded_and_chunk_decoded():
    a = make_high_level_adapter(SPEC, "openpi_flat")
    s = a.encode_state(reading(), est())
    assert s.shape == (1, 32) and torch.all(s[0, 14 + 2 + 2:] == 0)
    chunk = torch.rand(50, 32)
    cmds = a.decode_action(chunk)
    assert len(cmds) == 25
    c = cmds[3]
    assert torch.allclose(c.joint_targets["left_shoulder_pitch_joint"], chunk[3, 0:1])
    assert torch.allclose(c.gripper["left"], chunk[3, 14:15])
    assert "left_gripper_joint" in c.joint_targets
    assert torch.allclose(c.base_velocity, chunk[3:4, 16:19])


def test_openhlm_state_uses_jose_roll_pitch_and_skips_root_actions():
    a = make_high_level_adapter(SPEC, "openhlm_body")
    s = a.encode_state(reading(), est())
    assert s.shape == (1, 34)
    rpy = est().get("base_rpy")
    assert torch.allclose(s[0, :2], rpy[0, :2]) and torch.isclose(s[0, 2], torch.tensor(0.3))
    cmds = a.decode_action(torch.rand(50, 34))
    assert len(cmds) == 25 and len(cmds[0].joint_targets) == 31 and cmds[0].base_velocity is None


def test_groot_dict_roundtrip():
    a = make_high_level_adapter(SPEC, "groot_dict")
    s = a.encode_state(reading(), est())
    assert set(s) == {"state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"}
    raw = {"action.left_arm": torch.rand(16, 7), "action.right_arm": torch.rand(16, 7), "action.left_hand": torch.rand(16, 1),
           "action.right_hand": torch.rand(16, 1), "action.waist": torch.rand(16, 3), "action.base_velocity": torch.rand(16, 3)}
    cmds = a.decode_action(raw)
    assert len(cmds) == 16
    assert torch.allclose(cmds[5].joint_targets["waist_pitch_joint"], raw["action.waist"][5, 2:3])


def test_openvla_tokens_detokenize_to_ranges():
    a = make_high_level_adapter(SPEC, "openvla_tokens")
    assert a.encode_state(reading(), est()) is None
    cmd = a.decode_action(torch.tensor([255, 0, 128, 255, 0, 128, 255]))[0]
    d = cmd.ee_delta["right"][0]
    assert d[0] > 0.0199 and d[1] < -0.0199 and abs(float(d[2])) < 1e-3
    assert float(cmd.gripper["right"][0]) > 0.99


def test_sonic_latent_passthrough_and_gravity_state():
    a = make_high_level_adapter(SPEC, "sonic_latent")
    s = a.encode_state(reading(), est())
    assert s.shape == (1, 32) and torch.allclose(s[0, 29:], est().projected_gravity[0])
    latent = torch.randn(40, 64)
    cmds = a.decode_action(latent)
    assert len(cmds) == 40 and torch.allclose(cmds[7].latent[0], latent[7])


def test_wholebodyvla_flags_to_velocity():
    a = make_high_level_adapter(SPEC, "wholebodyvla_flags")
    n = len(SPEC.groups["upper"])
    raw = torch.zeros(n + 4)
    raw[n:n + 3] = torch.tensor([1.0, -1.0, 0.2])
    cmd = a.decode_action(raw)[0]
    assert torch.allclose(cmd.base_velocity, torch.tensor([[0.5, -0.3, 0.0]]))


def test_delta_mode_and_gripper_load():
    cfg = {"name": "t", "state": {"fields": [{"source": "gripper_load", "gripper": "left"}]},
           "action": {"fields": [{"kind": "joint_targets", "group": "left_arm", "mode": "delta"}]}}
    a = make_high_level_adapter(SPEC, cfg)
    r = reading()
    j = SPEC.joint_names.index("left_gripper_joint")
    assert torch.allclose(a.encode_state(r)[0, 0], r.effort[0, j] / SPEC.effort_limit[j])
    cmd = a.decode_action(torch.ones(7), r)[0]
    k = SPEC.joint_names.index("left_elbow_joint")
    assert torch.allclose(cmd.joint_targets["left_elbow_joint"], r.q[:, k] + 1)


def test_missing_estimate_and_bad_group_raise():
    with pytest.raises(ValueError):
        make_high_level_adapter(SPEC, "openhlm_body").encode_state(reading(), None)
    with pytest.raises(ValueError):
        make_high_level_adapter(load_embodiment("g1_29dof"), "openpi_flat")
