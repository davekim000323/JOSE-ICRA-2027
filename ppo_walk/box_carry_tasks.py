"""Gym registrations for the bimanual box-carry task and its estimator twin."""

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
