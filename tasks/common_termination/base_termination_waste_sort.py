
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_waste_sort(
    env: ManagerBasedRLEnv,
    carton_cfg: SceneEntityCfg = SceneEntityCfg("carton"),
    orange_cfg: SceneEntityCfg = SceneEntityCfg("orange"),
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    red_bin_cfg: SceneEntityCfg = SceneEntityCfg("red_bin"),
    green_bin_cfg: SceneEntityCfg = SceneEntityCfg("green_bin"),
    blue_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_bin"),
    bin_radius_xy: float = 0.22,
    bin_top_z_margin: float = 0.55,
) -> torch.Tensor:
    """任务成功判定：
    Carton -> RedBin
    BananaPeel -> GreenBin
    Bottle -> BlueBin
    判定规则：物体中心 XY 在桶中心半径内，Z 低于桶口上沿 + margin
    返回 [num_envs] bool tensor，True 表示应 reset
    """
    def _inside_bin(obj: RigidObject, bin_: RigidObject):
        p = obj.data.root_pos_w
        q = bin_.data.root_pos_w
        d_xy = torch.linalg.norm(p[:, :2] - q[:, :2], dim=-1)
        ok_xy = d_xy < bin_radius_xy
        ok_z = p[:, 2] < (q[:, 2] + bin_top_z_margin)
        return ok_xy & ok_z

    carton = env.scene[carton_cfg.name]
    orange = env.scene[orange_cfg.name]
    bottle = env.scene[bottle_cfg.name]
    red_bin = env.scene[red_bin_cfg.name]
    green_bin = env.scene[green_bin_cfg.name]
    blue_bin = env.scene[blue_bin_cfg.name]

    success = (
        # _inside_bin(carton, red_bin) &
        _inside_bin(orange, green_bin) &
        _inside_bin(bottle, blue_bin)
    )

    # 返回 True 表示 episode 结束（成功或失败都可以在外部额外加逻辑）
    return success


def reset_object_estimate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = -2.7,                # minimum x position threshold
    max_x: float = -2.2,                # maximum x position threshold
    min_y: float = -4.15,                # minimum y position threshold
    max_y: float = -3.55,                # maximum y position threshold
    min_height: float = 0.2,
) -> torch.Tensor:
    # when the object is not in the set return, reset
    # Get object entity from the scene
    # 1. get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    
    # Extract wheel position relative to environment origin
    # 2. get object position
    wheel_x = object.data.root_pos_w[:, 0]         # x position
    wheel_y = object.data.root_pos_w[:, 1]        # y position
    wheel_height = object.data.root_pos_w[:, 2]   # z position (height)
    done_x = (wheel_x < max_x) and  (wheel_x > min_x)
    done_y = (wheel_y < max_y) and (wheel_y > min_y)
    done_height = (wheel_height > min_height)
    done = done_x and done_y and done_height
    # print(f"done_x: {done_x}, done_y: {done_y}, done_height: {done_height}, done: {done}")
    # print(f"wheel_x: {wheel_x}, wheel_y: {wheel_y}, wheel_height: {wheel_height}")
    return  not done
