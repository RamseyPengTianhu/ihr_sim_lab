from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from tasks.common_observations.camera_state import get_camera_image
import torch

def get_items_world_pose(env):
    """返回三个垃圾物体的世界位姿 (pos+quat) 平铺"""
    def pose7(name: str):
        ent = env.scene[name]
        pos = ent.data.root_pos_w
        rot = ent.data.root_rot_w
        return torch.cat([pos, rot], dim=-1)
    return torch.cat([
        # pose7("carton"),
        pose7("orange"),
        pose7("bottle")
    ], dim=-1)

__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_dex3_joint_states",
    "get_camera_image",
    "get_items_world_pose"
]