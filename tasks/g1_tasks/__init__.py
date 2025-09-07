
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import
from . import pick_place_cylinder_g1_29dof_dex3
from . import pick_place_cylinder_g1_29dof_dex1
from . import pick_place_cylinder_g1_29dof_inspire

from . import pick_place_redblock_g1_29dof_dex1
from . import pick_place_redblock_g1_29dof_dex3
from . import stack_rgyblock_g1_29dof_dex1
from . import stack_rgyblock_g1_29dof_dex3
from . import stack_rgyblock_g1_29dof_inspire
from . import pick_redblock_into_drawer_g1_29dof_dex1
from . import pick_redblock_into_drawer_g1_29dof_dex3
from . import pick_place_redblock_g1_29dof_inspire
from . import waste_sort_g1_29dof_dex3   # ✅ 新增

# export all modules
__all__ = [
        "pick_place_cylinder_g1_29dof_dex3", "pick_place_cylinder_g1_29dof_dex1", 
        "pick_place_redblock_g1_29dof_dex1", "pick_place_redblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_dex1", "stack_rgyblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_inspire",
        "pick_redblock_into_drawer_g1_29dof_dex1","pick_redblock_into_drawer_g1_29dof_dex3",
        "pick_place_redblock_g1_29dof_inspire",
        "pick_place_cylinder_g1_29dof_inspire",
        "waste_sort_g1_29dof_dex3"   # ✅ 新增
]