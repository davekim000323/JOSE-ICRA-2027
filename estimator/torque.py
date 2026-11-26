"""Joint-torque channel for the JOSE estimator, read as a motor-current proxy.

Kinematics alone gives the base tilt relative to the stance surface. On a slope
that differs from the tilt relative to gravity, which is what the teacher reads
as ``projected_gravity``. The gravity loading term in the joint torques depends on
the tilt relative to gravity, so torque restores that observability.

On hardware the G1 motor drivers report current, and torque follows from the
torque constant. This module produces the simulated counterpart:

    tau_meas = g * clip(tau, -tau_max, tau_max) + b + n,    n ~ N(0, (sigma * tau_max)^2)

* ``g`` is a per-joint gain error (torque-constant and gear-efficiency mismatch),
  sampled once per episode from U[1 - gain_range, 1 + gain_range].
* ``b`` is a per-joint offset (current-sensor offset and unmodelled friction),
  sampled once per episode from N(0, (bias_std * tau_max)^2).
* ``n`` is white measurement noise, resampled every step.
* ``quantization`` (N) rounds to a fixed step, as a current ADC does.

The default magnitudes are assumptions chosen to be conservative, not values
identified on the G1. Identify them on hardware before quoting them as a sensor
model in the paper.

Two torque sources are available.

* ``pd`` reads ``ArticulationData.applied_torque``. For the implicit actuators
  the G1 asset uses, Isaac Lab fills this field with the explicit PD law
  ``kp (q* - q) - kd qd`` evaluated on the commanded target, clipped to the
  effort limit. This matches what a position-controlled motor driver commands,
  which is what its current tracks.
* ``solver`` reads the generalized joint forces PhysX actually applied,
  ``root_physx_view.get_dof_projected_joint_forces()``. This includes the
  implicit-drive solve and is closer to a true torque sensor. It is used only
  when the PhysX view exposes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

TORQUE_SOURCES = ("pd", "solver")
TORQUE_NOISE_PRESETS = ("none", "default", "high")


@dataclass(frozen=True)
class TorqueNoiseCfg:
    gain_range: float = 0.0
    bias_std: float = 0.0
    noise_std: float = 0.0
    quantization: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def enabled(self) -> bool:
        return any((self.gain_range, self.bias_std, self.noise_std, self.quantization))


def torque_noise_preset(name: str) -> TorqueNoiseCfg:
    name = name.lower()
    if name == "none":
        return TorqueNoiseCfg()
    if name == "default":
        # 10 % gain error, 1 % offset and 2 % white noise relative to the effort
        # limit, 0.01 N m quantization.
        return TorqueNoiseCfg(gain_range=0.10, bias_std=0.01, noise_std=0.02, quantization=0.01)
    if name == "high":
        return TorqueNoiseCfg(gain_range=0.20, bias_std=0.03, noise_std=0.05, quantization=0.05)
    raise ValueError(f"Unknown torque noise preset {name!r}; choose from {TORQUE_NOISE_PRESETS}")


def _effort_limits(data, num_joints: int, device, dtype) -> torch.Tensor:
    for name in ("joint_effort_limits", "joint_effort_limit", "soft_joint_effort_limits"):
        value = getattr(data, name, None)
        if value is not None:
            return value.to(device=device, dtype=dtype).abs().clamp_min(1.0e-6)
    return torch.ones((1, num_joints), device=device, dtype=dtype)


def read_joint_torque(core_env, source: str = "pd") -> torch.Tensor:
    """Raw simulated joint torque, (num_envs, num_joints), in N m."""
    robot = core_env.robot
    if source == "pd":
        data = robot.data
        for name in ("applied_torque", "computed_torque"):
            value = getattr(data, name, None)
            if value is not None:
                return value
        raise RuntimeError("Articulation exposes neither applied_torque nor computed_torque")
    if source == "solver":
        view = getattr(robot, "root_physx_view", None)
        getter = getattr(view, "get_dof_projected_joint_forces", None)
        if getter is None:
            raise RuntimeError("PhysX view does not expose get_dof_projected_joint_forces; use --torque-source pd")
        return getter()
    raise ValueError(f"Unknown torque source {source!r}; choose from {TORQUE_SOURCES}")


class TorqueSensorModel:
    """Applies the per-episode gain and offset and the per-step noise.

    Episode boundaries are detected from ``episode_length_buf == 0``, the same
    signal the PPO walk adapter uses to refill its estimate ring, so every
    rollout loop in the pipeline gets resampled parameters without extra hooks.
    The output is divided by the effort limit, so every joint enters the
    estimator on a comparable scale before the running normalizer.
    """

    def __init__(self, core_env, joint_ids: tuple[int, ...], source: str, noise: TorqueNoiseCfg):
        if source not in TORQUE_SOURCES:
            raise ValueError(f"Unknown torque source {source!r}; choose from {TORQUE_SOURCES}")
        self.core_env = core_env
        self.joint_ids = tuple(joint_ids)
        self.source = source
        self.noise = noise
        self._ids: torch.Tensor | None = None
        self._limit: torch.Tensor | None = None
        self._gain: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None

    def _sample(self, count: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        width = like.shape[-1]
        gain = 1.0 + self.noise.gain_range * (2.0 * torch.rand((count, width), device=like.device) - 1.0)
        bias = self.noise.bias_std * torch.randn((count, width), device=like.device)
        return gain.to(like.dtype), bias.to(like.dtype)

    def read(self) -> torch.Tensor:
        torque = read_joint_torque(self.core_env, self.source)
        if self._ids is None or self._ids.device != torque.device:
            self._ids = torch.as_tensor(self.joint_ids, device=torque.device)
        torque = torque.index_select(1, self._ids)
        if self._limit is None:
            limits = _effort_limits(self.core_env.robot.data, torque.shape[-1], torque.device, torque.dtype)
            if limits.shape[-1] != torque.shape[-1]:
                limits = limits.index_select(-1, self._ids)
            self._limit = limits
        limit = self._limit
        torque = torch.maximum(torch.minimum(torque, limit), -limit)
        if self.noise.enabled:
            num_envs = torque.shape[0]
            if self._gain is None or self._gain.shape[0] != num_envs:
                self._gain, self._bias = self._sample(num_envs, torque)
            else:
                just_reset = getattr(self.core_env, "episode_length_buf", None)
                if just_reset is not None:
                    just_reset = just_reset == 0
                    if bool(just_reset.any()):
                        gain, bias = self._sample(int(just_reset.sum()), torque)
                        self._gain[just_reset] = gain
                        self._bias[just_reset] = bias
            torque = self._gain * torque + limit * self._bias
            if self.noise.noise_std:
                torque = torque + limit * self.noise.noise_std * torch.randn_like(torque)
            if self.noise.quantization:
                torque = torch.round(torque / self.noise.quantization) * self.noise.quantization
        return torque / limit
