import math

import numpy as np
import torch

from modular.frames import (
    SensorChain, euler_yaw_rate, gravity_from_quat, matrix_from_quat, matrix_from_rpy,
    quat_from_matrix, rpy_from_gravity_yaw,
)

from helpers import (
    np_quat_to_matrix, random_quat, unitree_gravity_orientation, unitree_transform_imu_data,
)

G = torch.Generator().manual_seed(0)


def test_gravity_matches_unitree_helper():
    for q in random_quat(50, G):
        ours = gravity_from_quat(q.unsqueeze(0))[0].numpy()
        np.testing.assert_allclose(ours, unitree_gravity_orientation(q.numpy()), atol=1e-12)


def test_quat_matrix_round_trip():
    q = random_quat(100, G)
    np.testing.assert_allclose(quat_from_matrix(matrix_from_quat(q)).numpy(), q.numpy(), atol=1e-9)


def test_rpy_from_gravity_inverts_matrix_from_rpy():
    rpy = (torch.rand((200, 3), generator=G, dtype=torch.float64) - 0.5) * torch.tensor([2.0, 2.0, 6.0], dtype=torch.float64)
    rpy[:, 1] = rpy[:, 1].clamp(-1.3, 1.3)
    gravity = (matrix_from_rpy(rpy).transpose(-1, -2) @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
    rec = rpy_from_gravity_yaw(gravity, rpy[:, 2])
    np.testing.assert_allclose(rec.numpy(), rpy.numpy(), atol=1e-9)


def test_h1_torso_chain_matches_unitree_transform_imu_data():
    """Our sensor_to_base on the H1 chain equals Unitree's own torso -> pelvis step."""
    chain = SensorChain.from_dict({"frame": "torso", "chain": [{"joint": "torso_joint", "axis": [0, 0, 1]}]})
    for i in range(30):
        quat = random_quat(1, G)
        omega = torch.randn((1, 3), generator=G, dtype=torch.float64)
        yaw = torch.rand((1, 1), generator=G, dtype=torch.float64) * 2 - 1
        yaw_rate = torch.randn((1, 1), generator=G, dtype=torch.float64)
        r_pelvis, w_ref = unitree_transform_imu_data(float(yaw), float(yaw_rate), quat[0].numpy(), omega[0].numpy())
        g_ref = r_pelvis.T @ np.array([0.0, 0.0, -1.0])
        g_b, w_b = chain.sensor_to_base(gravity_from_quat(quat), omega, yaw, yaw_rate)
        np.testing.assert_allclose(g_b[0].numpy(), g_ref, atol=1e-9)
        np.testing.assert_allclose(w_b[0].numpy(), w_ref, atol=1e-9)


def test_chain_round_trip_three_joint_waist():
    chain = SensorChain.from_dict({"chain": [
        {"joint": "waist_yaw_joint", "axis": [0, 0, 1]},
        {"joint": "waist_roll_joint", "axis": [1, 0, 0]},
        {"joint": "waist_pitch_joint", "axis": [0, 1, 0]},
    ], "mount_rpy": [0.1, -0.2, 0.3]})
    g = torch.nn.functional.normalize(torch.randn((20, 3), generator=G, dtype=torch.float64), dim=-1)
    w = torch.randn((20, 3), generator=G, dtype=torch.float64)
    q = torch.randn((20, 3), generator=G, dtype=torch.float64)
    dq = torch.randn((20, 3), generator=G, dtype=torch.float64)
    gs, ws = chain.base_to_sensor(g, w, q, dq)
    gb, wb = chain.sensor_to_base(gs, ws, q, dq)
    np.testing.assert_allclose(gb.numpy(), g.numpy(), atol=1e-9)
    np.testing.assert_allclose(wb.numpy(), w.numpy(), atol=1e-9)


def test_sensor_quat_consistent_with_sensor_gravity():
    chain = SensorChain.from_dict({"chain": [{"joint": "torso_joint", "axis": [0, 0, 1]}]})
    base = random_quat(10, G)
    q = torch.randn((10, 1), generator=G, dtype=torch.float64)
    gs_from_quat = gravity_from_quat(chain.base_to_sensor_quat(base, q))
    gs, _ = chain.base_to_sensor(gravity_from_quat(base), torch.zeros((10, 3), dtype=torch.float64), q, q * 0)
    np.testing.assert_allclose(gs_from_quat.numpy(), gs.numpy(), atol=1e-9)


def test_euler_yaw_rate_on_tilted_body():
    """A body tilted 0.3 rad in roll spinning about world z at 0.5 rad/s."""
    roll, rate = 0.3, 0.5
    rpy = torch.tensor([[roll, 0.0, 0.0]], dtype=torch.float64)
    r = matrix_from_rpy(rpy)[0]
    omega_body = (r.T @ torch.tensor([0.0, 0.0, rate], dtype=torch.float64)).unsqueeze(0)
    gravity = (r.T @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)).unsqueeze(0)
    assert math.isclose(float(euler_yaw_rate(gravity, omega_body)), rate, rel_tol=1e-9)
