"""Robot-specific tasks registered with the existing Isaac Lab/Gym machinery."""
def register():
    import gymnasium as gym
    if "RobotLab-PipDrive-v0" not in gym.registry:
        gym.register(
            id="RobotLab-PipDrive-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "robotlab.config:PipEnvCfg",
                "rsl_rl_cfg_entry_point": "robotlab.config:PipRunnerCfg",
                "default_agent": "rsl_rl",
            },
        )

