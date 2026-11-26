"""Gripper lift environment exposing JOSE's estimator interface (11-D target)."""

from __future__ import annotations

import torch

from .mdp import gripper_lift_mdp as grip
from .walk_estimator_env import G1WalkEstimatorEnv


class G1GripperLiftEstimatorEnv(G1WalkEstimatorEnv):
    def get_estimator_target(self) -> torch.Tensor:
        data = self.robot.data
        return torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                grip.grip_force_obs(self),
                grip.payload_mass_obs(self),
            ),
            dim=-1,
        )

    def get_distillation_sensor_state(self) -> dict[str, torch.Tensor]:
        data = self.robot.data
        return {
            "joint_position": data.joint_pos,
            "joint_velocity": data.joint_vel,
            "previous_action": self.action_manager.action,
            "quaternion_wxyz": data.root_quat_w,
            "angular_velocity": data.root_ang_vel_b,
            "projected_gravity": data.projected_gravity_b,
            "object_pos_b": grip.object_pos_b(self),
            "object_axis_b": grip.object_axis_b(self),
            "target_pos_b": grip.target_pos_b(self),
        }

    def task_success(self) -> torch.Tensor:
        """1 where the grasped, lifted object is within 5 cm of the target."""
        return grip.task_success(self)
