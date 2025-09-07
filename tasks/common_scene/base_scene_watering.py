from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, ArticulationCfg

from tasks.common_config.robot_configs import G1RobotPresets, CameraPresets

class WateringSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(0.0, -1.0, 0.0), init_rot=(1, 0, 0, 0)
    )
    front_camera = CameraPresets.g1_front_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

    kettle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Props/Kettle",
        usd_path="REPLACE/with/kettle_2p3L.usd",
        semantics=["kettle"], collision=True
    )
    faucet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Props/Faucet",
        usd_path="REPLACE/with/faucet.usd",
        semantics=["faucet"], collision=True
    )
    faucet_head = RigidObjectCfg(   # 便于定位出水点
        prim_path="{ENV_REGEX_NS}/Props/FaucetHead",
        usd_path="REPLACE/with/faucet_head.usd",
        semantics=["faucet_head"], collision=False
    )
    flowerbed = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Props/FlowerBed",
        usd_path="REPLACE/with/flower_bed.usd",
        semantics=["flowerbed"], collision=True
    )
    # 多朵玫瑰（会变色的材质）
    rose = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Props/Rose_*",
        usd_path="REPLACE/with/paper_rose.usd",
        semantics=["rose"], collision=False
    )