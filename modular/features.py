"""Estimator input features, native per robot or canonical across robots.

``native`` reproduces the input the existing JOSE checkpoints were trained on.
It is the estimator joints' positions, then their velocities, then (if the
torque channel is on) their torques divided by the effort limit. With the G1
29-DoF spec and every joint selected, the column order equals
``estimator/adapters.py`` with ``use_torque`` on, so a G1 checkpoint loads as is.

``canonical`` maps every robot onto one fixed vocabulary of humanoid joint slots.
Each slot carries the position offset from the default pose, the velocity, the
effort-normalized torque and a presence mask. A robot without a given joint
leaves its slot at zero with mask 0. One network trained on several robots can
then read any of them, which is the cross-embodiment route in the survey
(section 7.3, option 3). Normalizing torque by the effort limit removes the
gearing and motor size differences between robots, which is why raw current is
never fed to the canonical layout.
"""

from __future__ import annotations

import torch

from .embodiment import EmbodimentSpec
from .frames import select
from .signals import JointReading

_SIDE = (
    "hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll",
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw", "gripper",
)
CANONICAL_SLOTS: tuple[str, ...] = (
    *(f"left_{name}" for name in _SIDE),
    *(f"right_{name}" for name in _SIDE),
    "waist_yaw", "waist_roll", "waist_pitch",
    "head_yaw", "head_pitch",
)
LAYOUTS = ("native", "canonical")


class FeatureBuilder:
    def __init__(self, spec: EmbodimentSpec, layout: str = "native", use_torque: bool = False):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if use_torque and spec.torque_source == "none":
            raise ValueError(f"{spec.name} reports no torque or current, cannot build a torque channel")
        self.spec = spec
        self.layout = layout
        self.use_torque = bool(use_torque)
        self._ids = spec.estimator_ids
        self._limit = torch.tensor(spec.effort_limit).clamp_min(1.0e-6)
        self._default = torch.tensor(spec.default_q)
        if layout == "canonical":
            unknown = sorted(set(spec.slots.values()) - set(CANONICAL_SLOTS))
            if unknown:
                raise ValueError(f"{spec.name}: slots {unknown} are not canonical")
            # Only estimator joints with a slot enter. A joint without a slot is
            # dropped rather than guessed, and ``unmapped_joints`` reports it.
            pairs = [(j, spec.slots[j]) for j in spec.estimator_joints if j in spec.slots]
            self._slot_joint_ids = [spec.joint_names.index(j) for j, _ in pairs]
            self._slot_index = [CANONICAL_SLOTS.index(s) for _, s in pairs]
            if len(set(self._slot_index)) != len(self._slot_index):
                raise ValueError(f"{spec.name}: two joints map to one canonical slot")

    @property
    def unmapped_joints(self) -> list[str]:
        if self.layout != "canonical":
            return []
        return [j for j in self.spec.estimator_joints if j not in self.spec.slots]

    @property
    def input_dim(self) -> int:
        if self.layout == "native":
            return len(self._ids) * (3 if self.use_torque else 2)
        return len(CANONICAL_SLOTS) * (4 if self.use_torque else 3)

    def _torque_norm(self, reading: JointReading) -> torch.Tensor:
        torque = self.spec.joint_torque(reading.effort)
        if torque is None:
            raise ValueError("Torque channel is on but the reading has no effort")
        limit = self._limit.to(torque)
        return torch.maximum(torch.minimum(torque, limit), -limit) / limit

    def __call__(self, reading: JointReading) -> torch.Tensor:
        if reading.q.shape[-1] != self.spec.num_joints:
            raise ValueError(f"{self.spec.name} expects {self.spec.num_joints} joints, got {reading.q.shape[-1]}")
        if self.layout == "native":
            parts = [select(reading.q, self._ids), select(reading.dq, self._ids)]
            if self.use_torque:
                parts.append(select(self._torque_norm(reading), self._ids))
            return torch.cat(parts, -1)

        batch, n_slots = reading.batch, len(CANONICAL_SLOTS)
        ids = self._slot_joint_ids
        slots = torch.as_tensor(self._slot_index, dtype=torch.long, device=reading.q.device)

        def scatter(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((batch, n_slots), dtype=reading.q.dtype, device=reading.q.device)
            out[:, slots] = select(values, ids)
            return out

        offset = reading.q - self._default.to(reading.q)
        mask = torch.zeros((batch, n_slots), dtype=reading.q.dtype, device=reading.q.device)
        mask[:, slots] = 1.0
        parts = [scatter(offset), scatter(reading.dq)]
        if self.use_torque:
            parts.append(scatter(self._torque_norm(reading)))
        parts.append(mask)
        return torch.cat(parts, -1)
