"""Frame algebra between the estimator base frame and the frame a controller reads.

JOSE always estimates in the pelvis frame. Controllers read the IMU where the
vendor mounted it. The G1 IMU sits in the pelvis. The H1 and H1-2 IMU sits in the
torso, above the waist yaw joint, and ``unitree_rl_gym`` rotates it back to the
pelvis with ``transform_imu_data``. A ``SensorChain`` makes that rotation a
property of the embodiment config, so the same estimate can feed a controller
that expects either frame.

Conventions. Quaternions are (w, x, y, z). ``R_rel`` maps a vector expressed in
the sensor frame to the base frame, ``v_base = R_rel v_sensor``, and is the
product of the joint rotations along the chain times a fixed mount rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation, axis (3,), angle (B,) -> (B, 3, 3)."""
    axis = torch.nn.functional.normalize(axis.to(angle.dtype), dim=-1)
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1.0 - c
    return torch.stack(
        (
            torch.stack((t * x * x + c, t * x * y - s * z, t * x * z + s * y), -1),
            torch.stack((t * x * y + s * z, t * y * y + c, t * y * z - s * x), -1),
            torch.stack((t * x * z - s * y, t * y * z + s * x, t * z * z + c), -1),
        ),
        -2,
    )


def matrix_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    """ZYX Euler, (B, 3) -> (B, 3, 3), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy.unbind(-1)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=rpy.dtype, device=rpy.device)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=rpy.dtype, device=rpy.device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=rpy.dtype, device=rpy.device)
    return axis_angle_matrix(ez, y) @ axis_angle_matrix(ey, p) @ axis_angle_matrix(ex, r)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 4) wxyz with non-negative w."""
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1.0e-12)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) / 2.0
    x = torch.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = torch.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = torch.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    q = torch.stack((w, x, y, z), -1)
    return torch.nn.functional.normalize(q, dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def gravity_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Projected gravity R^T (0, 0, -1), identical to unitree_rl_gym get_gravity_orientation."""
    down = torch.tensor([0.0, 0.0, -1.0], dtype=q.dtype, device=q.device)
    return (matrix_from_quat(q).transpose(-1, -2) @ down).squeeze(-1)


def rpy_from_gravity_yaw(gravity: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Roll and pitch from projected gravity, yaw passed through. (B, 3), (B,) -> (B, 3).

    With R = Rz Ry Rx, projected gravity is (sin p, -cos p sin r, -cos p cos r).
    """
    g = torch.nn.functional.normalize(gravity, dim=-1)
    pitch = torch.asin(g[:, 0].clamp(-1.0, 1.0))
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    return torch.stack((roll, pitch, yaw), -1)


def euler_yaw_rate(gravity: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
    """ZYX yaw rate from body angular velocity, (sin r w_y + cos r w_z) / cos p."""
    rpy = rpy_from_gravity_yaw(gravity, torch.zeros_like(gravity[:, 0]))
    roll, pitch = rpy[:, 0], rpy[:, 1]
    cos_p = torch.cos(pitch).clamp_min(1.0e-3)
    return (torch.sin(roll) * ang_vel[:, 1] + torch.cos(roll) * ang_vel[:, 2]) / cos_p


@dataclass
class ChainLink:
    joint: str
    axis: tuple[float, float, float]


@dataclass
class SensorChain:
    """Joints between the pelvis and the IMU, in order, plus a fixed mount rotation."""

    links: list[ChainLink] = field(default_factory=list)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: str = "pelvis"

    @classmethod
    def from_dict(cls, data: dict | None) -> "SensorChain":
        data = data or {}
        links = [ChainLink(item["joint"], tuple(float(v) for v in item["axis"])) for item in data.get("chain", [])]
        return cls(links, tuple(float(v) for v in data.get("mount_rpy", (0.0, 0.0, 0.0))), data.get("frame", "pelvis"))

    @property
    def joints(self) -> list[str]:
        return [link.joint for link in self.links]

    @property
    def is_identity(self) -> bool:
        return not self.links and not any(self.mount_rpy)

    def _rotations(self, q_chain: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return R_rel and the cumulative rotation before each link."""
        batch = q_chain.shape[0]
        eye = torch.eye(3, dtype=q_chain.dtype, device=q_chain.device).expand(batch, 3, 3)
        current = eye
        before: list[torch.Tensor] = []
        for index, link in enumerate(self.links):
            before.append(current)
            axis = torch.tensor(link.axis, dtype=q_chain.dtype, device=q_chain.device)
            current = current @ axis_angle_matrix(axis, q_chain[:, index])
        mount = matrix_from_rpy(
            torch.tensor([self.mount_rpy], dtype=q_chain.dtype, device=q_chain.device).expand(batch, 3)
        )
        return current @ mount, before

    def _relative_rate(self, q_chain: torch.Tensor, dq_chain: torch.Tensor, before) -> torch.Tensor:
        rate = torch.zeros((q_chain.shape[0], 3), dtype=q_chain.dtype, device=q_chain.device)
        for index, link in enumerate(self.links):
            axis = torch.tensor(link.axis, dtype=q_chain.dtype, device=q_chain.device)
            axis = torch.nn.functional.normalize(axis, dim=-1)
            rate = rate + (before[index] @ axis).squeeze(-1) * dq_chain[:, index : index + 1]
        return rate

    def base_to_sensor(
        self, gravity_b: torch.Tensor, ang_vel_b: torch.Tensor, q_chain: torch.Tensor, dq_chain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pelvis-frame gravity and angular velocity -> the IMU frame."""
        if self.is_identity:
            return gravity_b, ang_vel_b
        r_rel, before = self._rotations(q_chain)
        rt = r_rel.transpose(-1, -2)
        gravity_s = (rt @ gravity_b.unsqueeze(-1)).squeeze(-1)
        omega_s = (rt @ (ang_vel_b + self._relative_rate(q_chain, dq_chain, before)).unsqueeze(-1)).squeeze(-1)
        return gravity_s, omega_s

    def sensor_to_base(
        self, gravity_s: torch.Tensor, ang_vel_s: torch.Tensor, q_chain: torch.Tensor, dq_chain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IMU-frame readings -> pelvis frame. Used to build training labels from a real IMU."""
        if self.is_identity:
            return gravity_s, ang_vel_s
        r_rel, before = self._rotations(q_chain)
        gravity_b = (r_rel @ gravity_s.unsqueeze(-1)).squeeze(-1)
        omega_b = (r_rel @ ang_vel_s.unsqueeze(-1)).squeeze(-1) - self._relative_rate(q_chain, dq_chain, before)
        return gravity_b, omega_b

    def base_to_sensor_quat(self, base_quat: torch.Tensor, q_chain: torch.Tensor) -> torch.Tensor:
        """World orientation of the IMU from the world orientation of the pelvis."""
        if self.is_identity:
            return base_quat
        r_rel, _ = self._rotations(q_chain)
        return quat_from_matrix(matrix_from_quat(base_quat) @ r_rel)


def select(values: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return values[:, :0]
    return values.index_select(1, torch.as_tensor(list(ids), device=values.device))
