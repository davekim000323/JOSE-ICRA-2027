"""Box-carry environment exposing JOSE's estimator interface.

The estimator target extends the walk task's 9-D base state with the two
quantities a bimanual squeeze makes visible in the arm torques, the per-hand
box contact force and the payload mass. Layout matches
``schema.BOX_CARRY_TARGET_NAMES``.
"""

from __future__ import annotations

import torch

from .mdp import box_carry_mdp as box
from .walk_estimator_env import G1WalkEstimatorEnv


class G1BoxCarryEstimatorEnv(G1WalkEstimatorEnv):
    def get_estimator_target(self) -> torch.Tensor:
        data = self.robot.data
        return torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                box.hand_contact_force_obs(self),
                box.payload_mass_obs(self),
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
            # Deployable task inputs, the box-carry analogue of the velocity command.
            "box_pos_b": box.box_pos_b(self),
            "box_axis_b": box.box_axis_b(self),
            "goal_pos_b": box.goal_pos_b(self),
        }

    def task_success(self) -> torch.Tensor:
        """1 where the lifted box is within the goal radius, the closed-loop task metric."""
        return box.delivered(self)
