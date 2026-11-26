import numpy as np
import torch

from modular.embodiment import load_embodiment
from modular.frames import gravity_from_quat
from modular.ports import make_port
from modular.signals import JointReading, StateEstimate

from helpers import unitree_gravity_orientation, unitree_transform_imu_data


def estimate(batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    grav = torch.nn.functional.normalize(torch.randn((batch, 3), generator=g) + torch.tensor([0, 0, -3.0]), dim=-1)
    return StateEstimate(
        projected_gravity=grav, base_ang_vel=torch.randn((batch, 3), generator=g),
        base_lin_vel=torch.randn((batch, 3), generator=g), yaw=torch.randn(batch, generator=g),
        contact={"left_hand": torch.rand(batch, generator=g), "right_hand": torch.rand(batch, generator=g)},
        payload=torch.rand(batch, generator=g) * 3,
    )


def reading(spec, batch=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    return JointReading(torch.randn((batch, spec.num_joints), generator=g) * 0.3,
                        torch.randn((batch, spec.num_joints), generator=g))


def test_unitree_rl_gym_layout():
    spec = load_embodiment("g1_29dof")
    est, r = estimate(), reading(spec)
    obs = np.arange(47, dtype=np.float32)
    out = make_port(spec, "unitree_rl_gym").write(obs.copy(), est, r)
    np.testing.assert_allclose(out[0:3], 0.25 * est.base_ang_vel[0].numpy(), rtol=1e-6)
    np.testing.assert_allclose(out[3:6], est.projected_gravity[0].numpy(), rtol=1e-6)
    np.testing.assert_array_equal(out[6:], obs[6:])


def test_booster_t1_order_is_reversed():
    spec = load_embodiment("t1")
    est, r = estimate(), reading(spec)
    out = make_port(spec, "booster_t1").write(torch.zeros(47), est, r)
    assert torch.allclose(out[0:3], est.projected_gravity[0])
    assert torch.allclose(out[3:6], est.base_ang_vel[0])


def test_ppo_walk_port_writes_exactly_the_schema_history_indices(schema):
    spec = load_embodiment("g1_29dof")
    port = make_port(spec, "jose_ppo_walk")
    assert sorted(port.written_indices) == sorted(schema.ppo_walk_history_target_indices())
    est, r = estimate(batch=3), reading(spec, batch=3)
    out = port.write(torch.zeros(3, schema.PPO_WALK_OBSERVATION_SCHEMA.policy_dim), est, r)
    ids = torch.tensor(schema.ppo_walk_history_target_indices())
    expected = torch.cat([est.base_lin_vel, est.base_ang_vel * 0.2, est.projected_gravity], -1).repeat(1, 5)
    assert torch.allclose(out[:, ids], expected)


def test_box_carry_port_matches_schema(schema):
    spec = load_embodiment("g1_29dof")
    port = make_port(spec, "jose_box_carry")
    assert sorted(port.written_indices) == sorted(schema.box_carry_history_target_indices())
    est, r = estimate(), reading(spec)
    out = port.write(torch.zeros(1, schema.BOX_CARRY_POLICY_DIM), est, r)
    newest = torch.tensor(schema.BOX_CARRY_OBSERVATION_SCHEMA.estimator_target_indices)
    scales = torch.tensor(schema.box_carry_target_scales())
    target = torch.cat([est.base_lin_vel, est.base_ang_vel, est.projected_gravity,
                        est.contact["left_hand"][:, None], est.contact["right_hand"][:, None], est.payload[:, None]], -1)
    assert torch.allclose(out[:, newest], target * scales)


def test_history_ring_after_steps_and_reset():
    spec = load_embodiment("g1_29dof")
    port = make_port(spec, "jose_ppo_walk")
    r = reading(spec)
    for k in range(3):
        est = estimate(seed=k)
        out = port.write(torch.zeros(1, 495), est, r)
    # gravity block starts at 30, 5 frames of 3, newest last
    assert torch.allclose(out[0, 42:45], est.projected_gravity[0])
    assert torch.allclose(out[0, 39:42], estimate(seed=1).projected_gravity[0])
    port.reset(torch.tensor([True]))
    out = port.write(torch.zeros(1, 495), estimate(seed=7), r)
    assert torch.allclose(out[0, 30:45], estimate(seed=7).projected_gravity[0].repeat(5))


def test_h1_imu_message_survives_unitree_preprocessing():
    """JOSE fills the torso IMU message, Unitree's unmodified transform recovers the pelvis estimate."""
    spec = load_embodiment("h1")
    port = make_port(spec, "imu_message")
    est = estimate(seed=3)
    r = reading(spec, seed=4)
    msg = {"imu_state": {"quaternion": [1, 0, 0, 0], "gyroscope": [0, 0, 0], "rpy": [0, 0, 0], "accelerometer": [0, 0, 0]}}
    port.write(msg, est, r)
    j = spec.joint_names.index("torso_joint")
    imu = msg["imu_state"]
    r_pelvis, w = unitree_transform_imu_data(float(r.q[0, j]), float(r.dq[0, j]),
                                             np.asarray(imu["quaternion"]), np.asarray(imu["gyroscope"]))
    gravity = r_pelvis.T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(gravity, est.projected_gravity[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(w, est.base_ang_vel[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(imu["accelerometer"], (-9.81 * gravity_from_quat(torch.tensor([imu["quaternion"]])))[0].numpy(), atol=1e-4)


def test_g1_imu_message_gravity_matches_unitree_helper():
    spec = load_embodiment("g1_29dof")
    est, r = estimate(seed=5), reading(spec)
    msg = {"imu_state": {"quaternion": [0.0] * 4, "gyroscope": [0.0] * 3, "rpy": [0.0] * 3, "accelerometer": [0.0] * 3}}
    make_port(spec, "imu_message").write(msg, est, r)
    np.testing.assert_allclose(unitree_gravity_orientation(np.asarray(msg["imu_state"]["quaternion"])),
                               est.projected_gravity[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(msg["imu_state"]["gyroscope"], est.base_ang_vel[0].numpy(), atol=1e-6)
    assert abs(msg["imu_state"]["rpy"][2] - float(est.yaw)) < 1e-5
