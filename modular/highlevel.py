"""High-level adapters. Encode proprioceptive state for a VLA, decode its action chunk.

Humanoid VLAs differ in three ways that matter here.

1. State format. Some take one flat vector padded to a fixed width (openpi
   style), some a dict of named groups (GR00T style), some none at all.
2. Action format. Flat chunks, dicts of chunks, discrete tokens (OpenVLA
   style) or a latent code that a whole-body controller decodes (GR00T N1.7 with
   SONIC).
3. Which IMU-derived terms they read. OpenHLM and Psi0 put torso roll and pitch
   into the VLA state, the GR00T N1.7 SONIC card lists projected gravity next to
   the 29 joints. With the IMU removed those terms come from JOSE.

One ``HighLevelAdapter`` class covers all of them through a YAML layout. The
presets in ``configs/highlevel`` are written from the papers' text. They are
style templates, not verified against each released checkpoint. Check the
checkpoint's own modality config before running it.

State field sources
  joints.q / joints.dq   a joint ``group`` of the embodiment
  gripper                closure in [0, 1] of the 1-DoF ``gripper``
  gripper_load           torque over effort limit of the 1-DoF ``gripper``, from current
  jose                   a ``StateEstimate`` ``field``, optionally sub-indexed by ``index``

Every state field may carry a ``name``, used as the key when the format is dict.
Dict-format action fields read ``raw[key]``.

Action field kinds
  joint_targets  absolute or delta, for a joint group
  gripper        closure in [0, 1], optional ``invert``
  base_velocity  vx, vy, wz
  loco_flags     ternary flags in {-1, 0, 1} times ``speed`` per axis
  torso_rpy, base_height, ee_delta, latent, skip
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .embodiment import CONFIG_DIR, EmbodimentSpec
from .frames import select
from .signals import JointReading, StateEstimate, WholeBodyCommand

_ACTION_DIMS = {"base_velocity": 3, "loco_flags": 3, "torso_rpy": 3, "base_height": 1, "ee_delta": 6, "gripper": 1}


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


class HighLevelAdapter:
    def __init__(self, spec: EmbodimentSpec, config: dict):
        self.spec = spec
        self.config = config
        self.name = config.get("name", "custom")
        self.state_format = config.get("state", {}).get("format", "flat")
        self.state_fields = config.get("state", {}).get("fields", [])
        self.state_pad = config.get("state", {}).get("pad_to")
        action = config["action"]
        self.action_format = action.get("format", "flat")
        self.horizon = int(action.get("horizon", 1))
        self.execute = int(action.get("execute", self.horizon))
        self.action_fields = action["fields"]
        self.action_pad = action.get("pad_to")
        self.bins = int(action.get("bins", 256))
        for f in self.state_fields + self.action_fields:
            group = f.get("group")
            if group is not None and group not in spec.groups:
                raise ValueError(f"{self.name}: embodiment {spec.name} has no joint group {group!r}")
            if f.get("source") in ("gripper", "gripper_load") or f.get("kind") == "gripper":
                if f["gripper"] not in {g.name for g in spec.grippers}:
                    raise ValueError(f"{self.name}: embodiment {spec.name} has no gripper {f['gripper']!r}")
        if self.action_format == "flat" and self.action_pad is not None and self.action_dim > self.action_pad:
            raise ValueError(f"{self.name}: action fields need {self.action_dim} dims, pad_to is {self.action_pad}")

    # ------------------------------------------------------------------ state
    def _state_value(self, f: dict, reading: JointReading, est: StateEstimate | None) -> torch.Tensor:
        source = f["source"]
        if source in ("joints.q", "joints.dq"):
            values = reading.q if source == "joints.q" else reading.dq
            return select(values, self.spec.ids(self.spec.groups[f["group"]]))
        if source == "gripper":
            g = self.spec.gripper(f["gripper"])
            return g.to_unit(select(reading.q, self.spec.ids([g.joint])))
        if source == "gripper_load":
            g = self.spec.gripper(f["gripper"])
            torque = self.spec.joint_torque(reading.effort)
            if torque is None:
                raise ValueError(f"{self.spec.name} reports no gripper current or torque")
            idx = self.spec.joint_names.index(g.joint)
            return select(torque, [idx]) / self.spec.effort_limit[idx]
        if source == "jose":
            if est is None:
                raise ValueError(f"{self.name} state reads JOSE field {f['field']!r} but no estimate was given")
            value = est.get(f["field"])
            if "index" in f:
                value = value[:, list(f["index"])]
            return value
        raise ValueError(f"Unknown state source {source!r}")

    def encode_state(self, reading: JointReading, est: StateEstimate | None = None):
        values = {f.get("name", f"f{i}"): self._state_value(f, reading, est) for i, f in enumerate(self.state_fields)}
        if self.state_format == "dict":
            return values
        if self.state_format == "none":
            return None
        flat = torch.cat(list(values.values()), -1) if values else reading.q[:, :0]
        if self.state_pad is not None:
            if flat.shape[-1] > self.state_pad:
                raise ValueError(f"{self.name}: state has {flat.shape[-1]} dims, pad_to is {self.state_pad}")
            flat = torch.nn.functional.pad(flat, (0, self.state_pad - flat.shape[-1]))
        return flat

    # ----------------------------------------------------------------- action
    def _dims(self, f: dict) -> int:
        if "dims" in f:
            return int(f["dims"])
        if f["kind"] == "joint_targets":
            return len(self.spec.groups[f["group"]])
        return _ACTION_DIMS[f["kind"]]

    @property
    def action_dim(self) -> int:
        return sum(self._dims(f) for f in self.action_fields)

    def _detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Bin index in [0, bins) -> bin center in [-1, 1] -> physical range.

        The VLA wrapper maps vocabulary ids to bin indices. This step only
        undoes the uniform binning and the per-dimension normalization.
        """
        centers = (tokens.float() + 0.5) / self.bins * 2.0 - 1.0
        cols, start = [], 0
        for f in self.action_fields:
            d = self._dims(f)
            low = torch.as_tensor(f.get("low", [-1.0] * d), dtype=torch.float32)
            high = torch.as_tensor(f.get("high", [1.0] * d), dtype=torch.float32)
            cols.append(0.5 * (centers[..., start : start + d] + 1.0) * (high - low) + low)
            start += d
        return torch.cat(cols, -1)

    def _split(self, raw) -> tuple[list[torch.Tensor | None], int]:
        """Return one (B, H, d) tensor per action field, None for skipped fields."""
        if self.action_format == "dict":
            parts: list[torch.Tensor | None] = []
            for f in self.action_fields:
                if f["kind"] == "skip":
                    parts.append(None)
                    continue
                value = _as_tensor(raw[f["key"]])
                if value.ndim == 2:
                    value = value.unsqueeze(0)
                parts.append(value)
            horizon = next(p for p in parts if p is not None).shape[1]
            return parts, horizon
        value = _as_tensor(raw)
        if value.ndim == 1:
            value = value.view(1, 1, -1)
        elif value.ndim == 2:
            # A 2-D input is one chunk (H, D) when the layout has a horizon, else a batch (B, D).
            value = value.unsqueeze(0) if self.horizon > 1 else value.unsqueeze(1)
        if self.action_format == "tokens":
            value = self._detokenize(value)
        if value.shape[-1] < self.action_dim:
            raise ValueError(f"{self.name}: action has {value.shape[-1]} dims, layout needs {self.action_dim}")
        parts, start = [], 0
        for f in self.action_fields:
            d = self._dims(f)
            parts.append(value[..., start : start + d])
            start += d
        return parts, value.shape[1]

    def decode_action(self, raw, reading: JointReading | None = None) -> list[WholeBodyCommand]:
        """Action chunk -> the first ``execute`` steps as ``WholeBodyCommand``s."""
        parts, horizon = self._split(raw)
        commands = []
        for t in range(min(horizon, self.execute)):
            cmd = WholeBodyCommand()
            for i, f in enumerate(self.action_fields):
                kind = f["kind"]
                if kind == "skip":
                    continue
                v = parts[i][:, t]
                if kind == "joint_targets":
                    joints = self.spec.groups[f["group"]]
                    if f.get("mode", "absolute") == "delta":
                        if reading is None:
                            raise ValueError(f"{self.name}: delta joint targets need the current reading")
                        v = v + select(reading.q, self.spec.ids(joints))
                    for j, name in enumerate(joints):
                        cmd.joint_targets[name] = v[:, j]
                elif kind == "gripper":
                    closure = v[:, 0]
                    if f.get("invert", False):
                        closure = 1.0 - closure
                    cmd.gripper[f["gripper"]] = closure.clamp(0.0, 1.0)
                    g = self.spec.gripper(f["gripper"])
                    cmd.joint_targets[g.joint] = g.from_unit(cmd.gripper[f["gripper"]])
                elif kind == "base_velocity":
                    cmd.base_velocity = v
                elif kind == "loco_flags":
                    speed = torch.as_tensor(f.get("speed", [0.5, 0.3, 0.5]), dtype=v.dtype)
                    cmd.base_velocity = torch.round(v).clamp(-1, 1) * speed
                elif kind == "torso_rpy":
                    cmd.torso_rpy = v
                elif kind == "base_height":
                    cmd.base_height = v[:, 0]
                elif kind == "ee_delta":
                    cmd.ee_delta[f["arm"]] = v
                elif kind == "latent":
                    cmd.latent = v
                else:
                    raise ValueError(f"Unknown action kind {kind!r}")
            commands.append(cmd)
        return commands


def list_high_level_presets() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "highlevel").glob("*.yaml"))


def make_high_level_adapter(spec: EmbodimentSpec, config: str | dict | Path) -> HighLevelAdapter:
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.suffix:
            path = CONFIG_DIR / "highlevel" / f"{config}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HighLevelAdapter(spec, config)
