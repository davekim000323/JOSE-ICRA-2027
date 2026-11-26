import numpy as np
import torch


def random_quat(n, generator=None):
    q = torch.randn((n, 4), generator=generator, dtype=torch.float64)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[:, :1] < 0, -q, q)


def np_rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def np_quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def unitree_gravity_orientation(q):
    """Verbatim port of unitree_rl_gym deploy/deploy_real/common/rotation_helper.py."""
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


def unitree_transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    """Port of unitree_rl_gym transform_imu_data, returns pelvis rotation matrix and omega."""
    rz = np_rz(waist_yaw)
    r_torso = np_quat_to_matrix(np.asarray(imu_quat))
    r_pelvis = r_torso @ rz.T
    w = rz @ np.asarray(imu_omega) - np.array([0.0, 0.0, waist_yaw_omega])
    return r_pelvis, w


class ConstantModel:
    """Stands in for a trained estimator. Returns fixed outputs and records inputs."""

    def __init__(self, outputs: torch.Tensor):
        self.outputs = outputs
        self.inputs = []

    def __call__(self, history):
        self.inputs.append(history.clone())
        return self.outputs.expand(history.shape[0], -1).clone()
