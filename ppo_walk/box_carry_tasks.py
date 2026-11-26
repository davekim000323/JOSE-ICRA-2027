"""Gym registrations for the box-carry and gripper-lift manipulation tasks and their estimator twins."""

import gymnasium as gym

_PKG = __name__.rsplit(".", 2)[0]

gym.register(
    id="Isaac-G1-Box-Carry-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.box_carry_env_cfg:G1BoxCarryEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.box_carry_env_cfg:G1BoxCarryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1BoxCarryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Box-Carry-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.box_carry_env:G1BoxCarryEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.box_carry_env_cfg:G1BoxCarryEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.box_carry_env_cfg:G1BoxCarryEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1BoxCarryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Gripper-Lift-JOSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.gripper_lift_env_cfg:G1GripperLiftEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.gripper_lift_env_cfg:G1GripperLiftPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1GripperLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Gripper-Lift-Estimator-JOSE-v0",
    entry_point=f"{_PKG}.ppo_walk.gripper_lift_env:G1GripperLiftEstimatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PKG}.ppo_walk.gripper_lift_env_cfg:G1GripperLiftEstimatorEnvCfg",
        "play_env_cfg_entry_point": f"{_PKG}.ppo_walk.gripper_lift_env_cfg:G1GripperLiftEstimatorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PKG}.ppo_walk.agents.rsl_rl_ppo_cfg:G1GripperLiftPPORunnerCfg",
    },
)
