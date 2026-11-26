"""Low-level ports. Where a frozen controller reads its IMU, JOSE writes instead.

Two kinds of port cover every controller checked so far.

``ObservationPort`` overwrites slots of a flat observation vector. Each field
names an estimate key, a start index, a scale, a per-frame history length and a
frame. Verified layouts ship as presets.

* ``unitree_rl_gym``  obs[0:3] = gyro * 0.25, obs[3:6] = gravity, pelvis frame
  (``deploy/deploy_real/deploy_real.py``, G1, H1, H1-2 configs).
* ``booster_t1``      obs[0:3] = gravity, obs[3:6] = gyro, both scale 1, trunk frame
  (``booster_gym/deploy/utils/policy.py``). The order is the reverse of Unitree.
* ``jose_ppo_walk`` and ``jose_box_carry``, the Isaac Lab manager layouts in
  ``schema.py``, with the full 5-frame history overwritten.

``ImuMessagePort`` overwrites the IMU fields of a low-state message, in the frame
where the vendor mounted the IMU. The controller's own pre-processing, such as
``transform_imu_data`` on the H1, then runs unchanged. This is the port for a
controller whose source is closed or must not be edited, which includes the
whole-body controllers under current humanoid VLAs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .embodiment import CONFIG_DIR, EmbodimentSpec
from .frames import gravity_from_quat, matrix_from_rpy, quat_from_matrix, rpy_from_gravity_yaw, select
from .signals import JointReading, StateEstimate

GRAVITY = 9.81


@dataclass
class PortField:
    keys: list[str]
    start: int
    scale: float = 1.0
    history: int = 1
    frame: str = "base"  # "base" (pelvis) or "sensor" (where the embodiment's IMU sits)

    @property
    def width(self) -> int:
        return 3 * sum(1 for k in self.keys if not k.startswith(("contact.", "payload"))) + sum(
            1 for k in self.keys if k.startswith(("contact.", "payload"))
        )


def _sensor_frame(spec: EmbodimentSpec, est: StateEstimate, reading: JointReading):
    ids = spec.imu_chain_ids
    return spec.imu.base_to_sensor(
        est.projected_gravity, est.base_ang_vel, select(reading.q, ids), select(reading.dq, ids)
    )


def _value(key: str, frame: str, spec, est: StateEstimate, reading, cache: dict) -> torch.Tensor:
    if frame == "sensor" and key in ("projected_gravity", "base_ang_vel"):
        if "sensor" not in cache:
            cache["sensor"] = _sensor_frame(spec, est, reading)
        return cache["sensor"][0 if key == "projected_gravity" else 1]
    return est.get(key)


class ObservationPort:
    def __init__(self, spec: EmbodimentSpec, fields: list[PortField], name: str = "custom"):
        self.spec = spec
        self.fields = fields
        self.name = name
        self._rings: dict[int, torch.Tensor] = {}
        spans = sorted((f.start, f.start + f.width * f.history) for f in fields)
        for (a0, a1), (b0, _) in zip(spans, spans[1:]):
            if b0 < a1:
                raise ValueError(f"Port {name!r} fields overlap at index {b0}")

    @property
    def written_indices(self) -> list[int]:
        out: list[int] = []
        for f in self.fields:
            out.extend(range(f.start, f.start + f.width * f.history))
        return out

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self._rings.clear()
        else:
            for ring in self._rings.values():
                ring[mask] = float("nan")  # refilled from the next estimate

    def _push(self, index: int, value: torch.Tensor, history: int) -> torch.Tensor:
        ring = self._rings.get(index)
        if ring is None or ring.shape[0] != value.shape[0]:
            ring = value.unsqueeze(1).repeat(1, history, 1)
        else:
            ring = torch.roll(ring, -1, dims=1)
            ring[:, -1] = value
            stale = torch.isnan(ring).any(dim=-1).any(dim=-1)
            if bool(stale.any()):
                # Matches the Isaac Lab observation manager, which repopulates the
                # whole history with the first frame after a reset.
                ring[stale] = value[stale].unsqueeze(1)
        self._rings[index] = ring
        return ring.reshape(value.shape[0], -1)

    def write(self, obs, est: StateEstimate, reading: JointReading):
        """Overwrite the estimate slots of ``obs`` and return it. Accepts torch or numpy."""
        as_numpy = isinstance(obs, np.ndarray)
        target = torch.from_numpy(obs) if as_numpy else obs
        squeeze = target.ndim == 1
        if squeeze:
            target = target.unsqueeze(0)
        cache: dict = {}
        for index, f in enumerate(self.fields):
            value = torch.cat([_value(k, f.frame, self.spec, est, reading, cache) for k in f.keys], -1) * f.scale
            block = self._push(index, value, f.history) if f.history > 1 else value
            target[:, f.start : f.start + block.shape[-1]] = block.to(target.dtype)
        if squeeze:
            target = target.squeeze(0)
        return target.numpy() if as_numpy else target


class ImuMessagePort:
    """Fill ``msg.imu_state`` (or a dict) like the vendor IMU would.

    quaternion   wxyz, roll and pitch from the estimate, yaw from integration.
    gyroscope    rad/s in the IMU frame.
    rpy          roll, pitch, yaw of the IMU frame.
    accelerometer  quasi-static specific force, -9.81 times projected gravity.
                 It omits linear acceleration. Controllers that read it need a
                 dynamic model, which is listed as future work, not claimed.
    """

    name = "imu_message"

    def __init__(self, spec: EmbodimentSpec, attribute: str = "imu_state"):
        self.spec = spec
        self.attribute = attribute

    def reset(self, mask=None) -> None:
        pass

    def fields(self, est: StateEstimate, reading: JointReading) -> dict[str, torch.Tensor]:
        ids = self.spec.imu_chain_ids
        q_chain, dq_chain = select(reading.q, ids), select(reading.dq, ids)
        yaw = est.yaw if est.yaw is not None else torch.zeros_like(est.projected_gravity[:, 0])
        base_quat = quat_from_matrix(matrix_from_rpy(rpy_from_gravity_yaw(est.projected_gravity, yaw)))
        quat = self.spec.imu.base_to_sensor_quat(base_quat, q_chain)
        gravity_s, omega_s = self.spec.imu.base_to_sensor(est.projected_gravity, est.base_ang_vel, q_chain, dq_chain)
        sensor_rpy = rpy_from_gravity_yaw(gravity_from_quat(quat), torch.zeros_like(yaw))
        # Sensor yaw follows from the quaternion, gravity alone cannot give it.
        w, x, y, z = quat.unbind(-1)
        sensor_rpy[:, 2] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return {
            "quaternion": quat,
            "gyroscope": omega_s,
            "rpy": sensor_rpy,
            "accelerometer": -GRAVITY * gravity_s,
        }

    def write(self, msg: Any, est: StateEstimate, reading: JointReading, env: int = 0):
        values = {k: v[env].tolist() for k, v in self.fields(est, reading).items()}
        holder = msg[self.attribute] if isinstance(msg, dict) else getattr(msg, self.attribute)
        for key, value in values.items():
            if isinstance(holder, dict):
                holder[key] = value
            else:
                current = getattr(holder, key)
                # Keep the message's container type (list, array or DDS sequence).
                for i, v in enumerate(value):
                    current[i] = v
        return msg


def list_port_presets() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "ports").glob("*.yaml")) + ["imu_message"]


def make_port(spec: EmbodimentSpec, config: str | dict | Path):
    if config == "imu_message":
        return ImuMessagePort(spec)
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.suffix:
            path = CONFIG_DIR / "ports" / f"{config}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("type") == "imu_message":
        return ImuMessagePort(spec, config.get("attribute", "imu_state"))
    fields = []
    for item in config["fields"]:
        keys = item["key"] if isinstance(item["key"], list) else [item["key"]]
        fields.append(PortField(keys, int(item["start"]), float(item.get("scale", 1.0)),
                                int(item.get("history", 1)), item.get("frame", "base")))
    return ObservationPort(spec, fields, config.get("name", "custom"))
