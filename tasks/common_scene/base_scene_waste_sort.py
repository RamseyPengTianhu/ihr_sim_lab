# 三色垃圾桶 + 三类目标物 + 可选干扰物
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg
import os
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

project_root = os.environ.get("PROJECT_ROOT")

@configclass
class WasteSortSceneCfg(InteractiveSceneCfg):
    # 房间
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # 房间中心点
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
        ),
    )


    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-1.5,-4.2,-0.2],   # initial position [x, y, z]
                                                rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
        )

    # # 桌子（台面高度要和下面物体 z 对齐）
    # table = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/PackingTable",   
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.8, 0.0, 0.0], rot=[1,0,0,0]),
    #     spawn=UsdFileCfg(usd_path=f"{project_root}/assets/objects/packing_table.usd"),
    # )

    # # —— 垃圾桶（固定）——
    # # 若不想被撞翻：把 rigid_props.kinematic_enabled=True 打开；想能推倒就保留动力学并设置质量/摩擦
    # # —— 三个“垃圾桶”：圆柱占位（半径/高度你可以调）——
 # 垃圾桶
# 蓝色桶
    blue_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/blue_bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -1.5, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/"
                    "SimReady_Containers_Shipping_01_NVD@10010/Assets/simready_content/"
                    "common_assets/props/tote_b03/tote_b03_blue.usd"),
        ),
    )

    # 绿色桶
    green_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/green_bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -2, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/"
                    "SimReady_Containers_Shipping_01_NVD@10010/Assets/simready_content/"
                    "common_assets/props/tote_b03/tote_b03_green.usd"),
        ),
    )

    # 黄色桶
    red_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -2.5, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/"
                    "SimReady_Containers_Shipping_01_NVD@10010/Assets/simready_content/"
                    "common_assets/props/tote_b03/tote_b03_red.usd"),
        ),
    )

    # 目标物

    carton = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/carton",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.5, -4.00, 0.84], rot=[1,0,0,0]),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/Residential/Food/Boxed/salt_box.usd"),
        ),
    )
    orange = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/orange",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.75, -4.10, 0.84], rot=[1,0,0,0]),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/SimReady_Furniture_Misc_01_NVD@10010/Assets/simready_content/common_assets/props/orange_02/orange_02.usd"),
        ),
    )


    bottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.25, -4.10, 0.84], rot=[1,0,0,0]),
        spawn=UsdFileCfg(
            usd_path=(f"{project_root}/assets/objects/"
                    "SimReady_Containers_Shipping_01_NVD@10010/Assets/simready_content/"
                    "common_assets/props/whitepackerbottle_a04/whitepackerbottle_a04.usd"),
        ),
    )

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 俯视相机（调好了视角）
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-0.1, 3.6, 1.6),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )