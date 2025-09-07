# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import gymnasium as gym
import os

from . import pick_redblock_into_drawer_g1_29dof_dex1_joint_env_cfg


gym.register(
    id="Isaac-Pick-Redblock-Into-Drawer-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_redblock_into_drawer_g1_29dof_dex1_joint_env_cfg.PickRedblockIntoDrawerG129DEX1BaseFixEnvCfg,
    },
    disable_env_checker=True,
)

