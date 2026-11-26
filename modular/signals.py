"""Plain data types that cross the layer boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JointReading:
    """One control tick of joint-side sensing, batched as (B, n_joints).

    ``effort`` holds whatever the drive reports, interpreted by the embodiment's
    torque source. For ``current`` it is motor current in A, for ``tau_est`` it is
    the vendor torque estimate in N m. ``None`` means the robot reports neither.
    Joint order is the embodiment's ``joint_names`` order.
    """

    q: torch.Tensor
    dq: torch.Tensor
    effort: torch.Tensor | None = None
    stamp: float = 0.0

    def __post_init__(self) -> None:
        if self.q.ndim == 1:
            self.q = self.q.unsqueeze(0)
        if self.dq.ndim == 1:
            self.dq = self.dq.unsqueeze(0)
        if self.effort is not None and self.effort.ndim == 1:
            self.effort = self.effort.unsqueeze(0)
        if self.q.shape != self.dq.shape:
            raise ValueError(f"q {tuple(self.q.shape)} and dq {tuple(self.dq.shape)} differ")
        if self.effort is not None and self.effort.shape != self.q.shape:
            raise ValueError(f"effort {tuple(self.effort.shape)} does not match q {tuple(self.q.shape)}")

    @property
    def batch(self) -> int:
        return self.q.shape[0]


@dataclass
class StateEstimate:
    """What JOSE recovers, always in the estimator base frame (the pelvis).

    Fields the loaded model does not predict stay ``None``. Ports and high-level
    adapters raise when a configuration asks for a field that is ``None``, so a
    missing output is never silently replaced by zeros.
    """

    projected_gravity: torch.Tensor  # (B, 3), unit norm
    base_ang_vel: torch.Tensor  # (B, 3), rad/s
    base_lin_vel: torch.Tensor | None = None  # (B, 3), m/s
    yaw: torch.Tensor | None = None  # (B,), integrated, drifts
    contact: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> (B,)
    payload: torch.Tensor | None = None  # (B,), kg
    raw: torch.Tensor | None = None  # (B, n_outputs), unparsed model output

    def get(self, key: str) -> torch.Tensor:
        """Resolve ``projected_gravity``, ``contact.left_hand`` and similar keys."""
        if key.startswith("contact."):
            name = key.split(".", 1)[1]
            if name not in self.contact:
                raise KeyError(f"Estimator does not output contact {name!r}; has {sorted(self.contact)}")
            return self.contact[name].unsqueeze(-1)
        if key == "base_rpy":
            from .frames import rpy_from_gravity_yaw

            yaw = self.yaw if self.yaw is not None else torch.zeros_like(self.projected_gravity[:, 0])
            return rpy_from_gravity_yaw(self.projected_gravity, yaw)
        value = getattr(self, key, None)
        if value is None:
            raise KeyError(f"Estimator does not output {key!r}")
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"{key!r} is not a tensor field")
        return value if value.ndim == 2 else value.unsqueeze(-1)


@dataclass
class WholeBodyCommand:
    """One step of a decoded high-level action, robot-agnostic.

    Any subset can be set. The stack forwards it to the low-level controller,
    which decides how to realize each part (joint PD, IK, locomotion policy).
    """

    joint_targets: dict[str, torch.Tensor] = field(default_factory=dict)  # joint name -> (B,)
    gripper: dict[str, torch.Tensor] = field(default_factory=dict)  # gripper name -> (B,), 0 open, 1 closed
    base_velocity: torch.Tensor | None = None  # (B, 3), vx vy wz
    torso_rpy: torch.Tensor | None = None  # (B, 3)
    base_height: torch.Tensor | None = None  # (B,)
    ee_delta: dict[str, torch.Tensor] = field(default_factory=dict)  # arm name -> (B, 6)
    latent: torch.Tensor | None = None  # (B, d), passed through to a latent-conditioned WBC
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
