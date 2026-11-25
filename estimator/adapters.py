"""Policy adapters that make AMP and PPO share the estimator pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..schema import (
    AMP_OBSERVATION_SCHEMA,
    BOX_CARRY_HISTORY_LENGTH,
    BOX_CARRY_OBSERVATION_SCHEMA,
    box_carry_history_target_indices,
    box_carry_target_scales,
    G1_KEY_BODY_NAMES,
    PPO_WALK_OBSERVATION_SCHEMA,
    ObservationSchema,
    joint_indices,
    ppo_walk_history_target_indices,
    ppo_walk_target_scales,
    PPO_WALK_HISTORY_LENGTH,
)
from ..skrl_compat import deterministic_action
from ..task_math import inject_observation_estimate
from .torque import TorqueNoiseCfg, TorqueSensorModel


def unwrap_direct_env(env):
    current = env
    for _ in range(32):
        candidate = getattr(current, "unwrapped", current)
        if hasattr(candidate, "get_estimator_joint_state") and hasattr(candidate, "get_estimator_target"):
            return candidate
        next_env = getattr(current, "_env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Environment does not implement the JOSE estimator interface")


class PolicyAdapter(ABC):
    schema: ObservationSchema

    #: Inertial channels appended to the estimator input when ``use_imu`` is set:
    #: body-frame angular velocity and the projected gravity vector, three each.
    IMU_INPUT_DIM = 6

    def __init__(
        self,
        env,
        joint_preset: str = "all",
        use_imu: bool = False,
        use_torque: bool = False,
        torque_source: str = "pd",
        torque_noise: TorqueNoiseCfg | None = None,
        use_action: bool = False,
    ):
        self.env = env
        self.core_env = unwrap_direct_env(env)
        _, _, names = self.core_env.get_estimator_joint_state()
        self.joint_preset = joint_preset
        self.joint_ids = joint_indices(names, joint_preset)
        self.use_imu = bool(use_imu)
        self.use_torque = bool(use_torque)
        self.torque_source = torque_source
        self.torque_noise = torque_noise or TorqueNoiseCfg()
        self.use_action = bool(use_action)
        self.torque_sensor = (
            TorqueSensorModel(self.core_env, self.joint_ids, torque_source, self.torque_noise)
            if self.use_torque
            else None
        )

    @property
    def input_dim(self) -> int:
        return (
            2 * len(self.joint_ids)
            + (self.IMU_INPUT_DIM if self.use_imu else 0)
            + (len(self.joint_ids) if self.use_torque else 0)
            + (self.schema.action_dim if self.use_action else 0)
        )

    def input_config(self) -> dict:
        """What the estimator reads, stored in checkpoints so loaders rebuild the same adapter."""
        return {
            "imu_input": self.use_imu,
            "torque_input": self.use_torque,
            "torque_source": self.torque_source,
            "torque_noise": self.torque_noise.to_dict(),
            "action_input": self.use_action,
        }

    def _last_action(self) -> torch.Tensor:
        manager = getattr(self.core_env, "action_manager", None)
        if manager is not None:
            return manager.action
        actions = getattr(self.core_env, "actions", None)
        if actions is None:
            raise RuntimeError("Environment exposes neither action_manager nor actions")
        return actions

    def estimator_input(self) -> torch.Tensor:
        # Deliberately use the simulator-provided joint velocity. There is no
        # finite-difference, encoder quantization, EMA, or hardware-noise path.
        joint_pos, joint_vel, _ = self.core_env.get_estimator_joint_state()
        ids = torch.as_tensor(self.joint_ids, device=joint_pos.device)
        joints = torch.cat(
            (joint_pos.index_select(1, ids), joint_vel.index_select(1, ids)), dim=-1
        )
        # Optional channels are appended after the joints in a fixed order
        # (IMU, torque, action), so the joint block keeps its indices in every arm.
        extras: list[torch.Tensor] = []
        if self.use_imu:
            sensors = self.core_env.get_distillation_sensor_state()
            extras += [sensors["angular_velocity"], sensors["projected_gravity"]]
        if self.use_torque:
            # Motor-current proxy, normalized by the effort limit (estimator/torque.py).
            extras.append(self.torque_sensor.read().to(joints.dtype))
        if self.use_action:
            # Control arm for the torque ablation. The PD torque is a function of
            # the commanded target, so encoder+action separates the information
            # carried by the load from the information carried by the command.
            extras.append(self._last_action().to(joints.dtype))
        if not extras:
            return joints
        return torch.cat((joints, *extras), dim=-1)

    def estimator_target(self) -> torch.Tensor:
        target = self.core_env.get_estimator_target()
        if target.shape[-1] != self.schema.estimator_target_dim:
            raise RuntimeError(
                f"Adapter expected target dim {self.schema.estimator_target_dim}, got {target.shape[-1]}"
            )
        return target

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if estimate.shape[-1] != self.schema.estimator_target_dim:
            raise ValueError("Estimator output does not match the policy schema")
        return inject_observation_estimate(observations, estimate, self.schema.estimator_target_indices)

    def tracked_body_ids(self) -> list[int]:
        """Indices of the bodies whose positions MPJPE is measured over.

        Prefers the 10 AMP key bodies so the numbers line up with the key-body
        convention the rest of the pipeline already uses; falls back to every
        body for envs that define no key bodies (ppo_walk).
        """
        names = list(self.core_env.robot.data.body_names)
        tracked = [names.index(name) for name in G1_KEY_BODY_NAMES if name in names]
        return tracked or list(range(len(names)))

    def body_names(self) -> list[str]:
        names = list(self.core_env.robot.data.body_names)
        return [names[index] for index in self.tracked_body_ids()]

    def body_positions(self) -> torch.Tensor:
        """World positions of the tracked bodies, (num_envs, num_tracked, 3)."""
        ids = torch.as_tensor(self.tracked_body_ids(), device=self.core_env.robot.data.body_pos_w.device)
        return self.core_env.robot.data.body_pos_w.index_select(1, ids)

    def root_position(self) -> torch.Tensor:
        """World position of the root body, (num_envs, 3)."""
        return self.core_env.robot.data.root_pos_w

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def action(self, agent: Any, observations: torch.Tensor) -> torch.Tensor:
        states = self.env.state() if hasattr(self.env, "state") else None
        return deterministic_action(agent, observations, states)


class AmpPolicyAdapter(PolicyAdapter):
    schema = AMP_OBSERVATION_SCHEMA

    def name(self) -> str:
        return "amp"


class PpoWalkPolicyAdapter(PolicyAdapter):
    """Adapter for the manager-based PPO walk teacher driven by rsl-rl.

    Two things differ from the Direct-workflow adapters and both are plumbing,
    not method:

    * The teacher is an rsl-rl policy, so it is called directly instead of going
      through SKRL's ``act`` API.
    * The policy observation carries a five-step history per term, and the
      observation manager stores ``base_ang_vel`` pre-scaled by 0.2. Injection
      therefore rescales the estimate and overwrites the *entire* history block
      of the three estimated terms. Replacing only the newest frame would leave
      four frames of ground-truth privileged state visible to the policy, which
      would understate the closed-loop cost of estimation error.
    """

    schema = PPO_WALK_OBSERVATION_SCHEMA

    def __init__(self, env, joint_preset: str = "all", use_imu: bool = False, **input_options):
        super().__init__(env, joint_preset, use_imu, **input_options)
        self._history_indices = ppo_walk_history_target_indices()
        self._scales: torch.Tensor | None = None
        self._ring: torch.Tensor | None = None

    def name(self) -> str:
        return "ppo_walk"

    def action(self, agent: Any, observations: torch.Tensor) -> torch.Tensor:
        return agent(self.env.as_policy_input(observations))

    def _scaled(self, estimate: torch.Tensor) -> torch.Tensor:
        if self._scales is None:
            self._scales = torch.tensor(
                ppo_walk_target_scales(), device=estimate.device, dtype=estimate.dtype
            )
        return estimate * self._scales

    def _push(self, estimate: torch.Tensor) -> torch.Tensor:
        """Roll the estimate ring and return it flattened oldest-frame-first.

        The ring is refilled with the current estimate for environments that just
        reset, matching how the observation manager repopulates its own history
        buffer after a reset.
        """
        scaled = self._scaled(estimate)
        if self._ring is None or self._ring.shape[0] != scaled.shape[0]:
            self._ring = scaled.unsqueeze(1).repeat(1, PPO_WALK_HISTORY_LENGTH, 1)
        else:
            self._ring = torch.roll(self._ring, -1, dims=1)
            self._ring[:, -1] = scaled
            just_reset = self.core_env.episode_length_buf == 0
            if bool(just_reset.any()):
                self._ring[just_reset] = scaled[just_reset].unsqueeze(1)
        return self._ring.reshape(scaled.shape[0], -1)

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if estimate.shape[-1] != self.schema.estimator_target_dim:
            raise ValueError("Estimator output does not match the policy schema")
        return inject_observation_estimate(observations, self._push(estimate), self._history_indices)


class BoxCarryPolicyAdapter(PpoWalkPolicyAdapter):
    """Box-carry teacher. Same rsl-rl plumbing as the walk, wider 12-D target.

    The target adds per-hand contact force and payload mass to the 9-D base
    state. Injection overwrites the full five-frame history of all five
    estimated terms, for the same reason as the walk adapter.
    """

    schema = BOX_CARRY_OBSERVATION_SCHEMA

    def __init__(self, env, joint_preset: str = "all", use_imu: bool = False, **input_options):
        super().__init__(env, joint_preset, use_imu, **input_options)
        self._history_indices = box_carry_history_target_indices()

    def name(self) -> str:
        return "box_carry"

    def _scaled(self, estimate: torch.Tensor) -> torch.Tensor:
        if self._scales is None:
            self._scales = torch.tensor(box_carry_target_scales(), device=estimate.device, dtype=estimate.dtype)
        return estimate * self._scales

    def _push(self, estimate: torch.Tensor) -> torch.Tensor:
        scaled = self._scaled(estimate)
        if self._ring is None or self._ring.shape[0] != scaled.shape[0]:
            self._ring = scaled.unsqueeze(1).repeat(1, BOX_CARRY_HISTORY_LENGTH, 1)
        else:
            self._ring = torch.roll(self._ring, -1, dims=1)
            self._ring[:, -1] = scaled
            just_reset = self.core_env.episode_length_buf == 0
            if bool(just_reset.any()):
                self._ring[just_reset] = scaled[just_reset].unsqueeze(1)
        return self._ring.reshape(scaled.shape[0], -1)


def make_policy_adapter(
    kind: str,
    env,
    joint_preset: str = "all",
    use_imu: bool = False,
    use_torque: bool = False,
    torque_source: str = "pd",
    torque_noise: TorqueNoiseCfg | None = None,
    use_action: bool = False,
) -> PolicyAdapter:
    """Build the adapter for ``kind``.

    ``use_imu`` defaults to False, the JOSE configuration: the estimator sees
    joint encoders and nothing else.
    Setting it adds the six inertial channels to the estimator's *input* only --
    the target, the injection indices and the policy observation are untouched.
    ``use_torque`` appends the joint-torque channel (estimator/torque.py) and
    ``use_action`` the last action, both on the input side only.
    """
    options = dict(
        use_torque=use_torque, torque_source=torque_source,
        torque_noise=torque_noise, use_action=use_action,
    )
    kind = kind.lower()
    if kind == "amp":
        return AmpPolicyAdapter(env, joint_preset, use_imu, **options)
    # "ppo" named the SKRL Direct PPO walk teacher, which was removed along with
    # its environment. It now resolves to the rsl-rl walk teacher so existing
    # commands and logged run metadata keep working.
    if kind in ("ppo_walk", "ppo"):
        return PpoWalkPolicyAdapter(env, joint_preset, use_imu, **options)
    if kind == "box_carry":
        return BoxCarryPolicyAdapter(env, joint_preset, use_imu, **options)
    raise ValueError(f"Unknown policy adapter {kind!r}; choose amp, ppo_walk or box_carry")
