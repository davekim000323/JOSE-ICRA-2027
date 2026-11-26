"""Embodiment specs, loaded from ``modular/configs/embodiments/*.yaml``.

A spec is everything JOSE needs to know about a robot and nothing about the policy
that drives it. Adding a robot means writing one YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from .frames import SensorChain

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TORQUE_SOURCES = ("current", "tau_est", "none")


def _per_joint(value: Any, names: list[str], key: str) -> list[float]:
    """Scalar, list in joint order, or {joint: value} with a ``default`` entry."""
    if value is None:
        return [1.0] * len(names)
    if isinstance(value, (int, float)):
        return [float(value)] * len(names)
    if isinstance(value, dict):
        default = value.get("default")
        out = []
        for name in names:
            if name in value:
                out.append(float(value[name]))
            elif default is not None:
                out.append(float(default))
            else:
                raise ValueError(f"{key} has no entry for {name!r} and no default")
        return out
    values = [float(v) for v in value]
    if len(values) != len(names):
        raise ValueError(f"{key} has {len(values)} entries for {len(names)} joints")
    return values


@dataclass
class GripperSpec:
    """A 1-DoF gripper. ``open`` and ``closed`` are joint positions in rad or m."""

    name: str
    joint: str
    open: float
    closed: float

    def to_unit(self, q: torch.Tensor) -> torch.Tensor:
        """Joint position -> closure in [0, 1], 0 open and 1 closed."""
        return ((q - self.open) / (self.closed - self.open)).clamp(0.0, 1.0)

    def from_unit(self, closure: torch.Tensor) -> torch.Tensor:
        return self.open + closure.clamp(0.0, 1.0) * (self.closed - self.open)


@dataclass
class EmbodimentSpec:
    name: str
    joint_names: list[str]
    control_dt: float
    estimator_joints: list[str]
    slots: dict[str, str]
    default_q: list[float]
    effort_limit: list[float]
    torque_source: str = "none"
    torque_constant: list[float] = field(default_factory=list)  # N m per A at the joint
    imu: SensorChain = field(default_factory=SensorChain)
    grippers: list[GripperSpec] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        names = self.joint_names
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate joint names")
        missing = [j for j in self.estimator_joints if j not in names]
        if missing:
            raise ValueError(f"{self.name}: estimator joints not in joint_names {missing}")
        for joint in self.imu.joints:
            if joint not in names:
                raise ValueError(f"{self.name}: IMU chain joint {joint!r} not in joint_names")
        for joint in self.slots:
            if joint not in names:
                raise ValueError(f"{self.name}: slot map names unknown joint {joint!r}")
        for gripper in self.grippers:
            if gripper.joint not in names:
                raise ValueError(f"{self.name}: gripper joint {gripper.joint!r} not in joint_names")
        for group, joints in self.groups.items():
            unknown = [j for j in joints if j not in names]
            if unknown:
                raise ValueError(f"{self.name}: group {group!r} names unknown joints {unknown}")
        if self.torque_source not in TORQUE_SOURCES:
            raise ValueError(f"{self.name}: torque_source must be one of {TORQUE_SOURCES}")
        if self.torque_source == "current" and len(self.torque_constant) != len(names):
            raise ValueError(f"{self.name}: torque_source current needs a torque_constant per joint")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def ids(self, joints: list[str]) -> list[int]:
        return [self.joint_names.index(j) for j in joints]

    @property
    def estimator_ids(self) -> list[int]:
        return self.ids(self.estimator_joints)

    @property
    def imu_chain_ids(self) -> list[int]:
        return self.ids(self.imu.joints)

    def gripper(self, name: str) -> GripperSpec:
        for gripper in self.grippers:
            if gripper.name == name:
                return gripper
        raise KeyError(f"{self.name} has no gripper {name!r}")

    def joint_torque(self, effort: torch.Tensor | None) -> torch.Tensor | None:
        """Drive reading -> joint torque in N m, or None when the robot reports neither."""
        if effort is None or self.torque_source == "none":
            return None
        if self.torque_source == "current":
            kt = torch.tensor(self.torque_constant, dtype=effort.dtype, device=effort.device)
            return effort * kt
        return effort

    @classmethod
    def from_dict(cls, data: dict) -> "EmbodimentSpec":
        names = list(data["joint_names"])
        estimator = data.get("estimator_joints", "all")
        if estimator == "all":
            estimator = list(names)
        elif isinstance(estimator, str):
            estimator = list(data["groups"][estimator])
        torque = data.get("torque", {}) or {}
        source = torque.get("source", "none")
        if source == "current" and torque.get("constant") is None:
            raise ValueError(f"{data['name']}: torque source current needs torque.constant (N m per A)")
        return cls(
            name=data["name"],
            joint_names=names,
            control_dt=float(data.get("control_dt", 0.02)),
            estimator_joints=list(estimator),
            slots=dict(data.get("slots", {})),
            default_q=_per_joint(data.get("default_q", 0.0), names, "default_q"),
            effort_limit=_per_joint(data.get("effort_limit"), names, "effort_limit"),
            torque_source=source,
            torque_constant=_per_joint(torque.get("constant"), names, "torque.constant") if source == "current" else [],
            imu=SensorChain.from_dict(data.get("imu")),
            grippers=[GripperSpec(**g) for g in data.get("grippers", [])],
            groups={k: list(v) for k, v in (data.get("groups") or {}).items()},
            notes=str(data.get("notes", "")),
        )


def list_embodiments() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "embodiments").glob("*.yaml"))


def load_embodiment(name_or_path: str | Path, overrides: dict | None = None) -> EmbodimentSpec:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / "embodiments" / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if overrides:
        data = {**data, **overrides}
    return EmbodimentSpec.from_dict(data)
