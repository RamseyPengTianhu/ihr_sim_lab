import gymnasium as gym

gym.register(
    id="Isaac-Waste-Sort-G129-Dex3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "tasks.g1_tasks.waste_sort_g1_29dof_dex3.waste_sort_g1_29dof_dex3_joint_env_cfg:WasteSortG1Dex3EnvCfg"
    },
    disable_env_checker=True,
)