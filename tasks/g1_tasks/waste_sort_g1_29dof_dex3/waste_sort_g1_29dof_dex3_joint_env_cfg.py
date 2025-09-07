# tasks/g1_tasks/waste_sort_g1_29dof_dex3/waste_sort_g1_29dof_dex3_joint_env_cfg.py
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from tasks.common_scene.base_scene_waste_sort import WasteSortSceneCfg
from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

from pink.tasks import FrameTask
from isaaclab.scene import InteractiveSceneCfg



import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp

@configclass
class ObjectTableSceneCfg(WasteSortSceneCfg):
    # ... 你的地面 / 房间 / 桶 / 物体 ...

    # 5) 机器人（名字必须叫 robot，和 ActionsCfg.asset_name 对齐）
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-1.5, -3.55, 0.8),
        init_rot=(0.7071, 0.0, 0.0, -0.7071)
    )

    # 6) 相机（按需开启）
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()




@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        robot_joint = ObsTerm(func=mdp.get_robot_boy_joint_states)  # ✅ 修正名
        # 如需位置观测可再加： items = ObsTerm(func=obs_mdp.get_items_world_pose)
        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False
    policy: Policy = Policy()
@configclass
class EventCfg:
    reset_scene = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    
    # reset_carton = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x":[0.5,0.9], "y":[-0.3,0.3], "z":[0.75,0.75]},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("carton"),
    #     },
    # )
    # reset_banana = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x":[0.5,0.9], "y":[-0.3,0.3], "z":[0.75,0.75]},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("banana"),
    #     },
    # )
    # reset_bottle = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x":[0.5,0.9], "y":[-0.3,0.3], "z":[0.75,0.75]},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("bottle"),
    #     },
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=mdp.reset_waste_sort)  # ✅ 对齐导出名

@configclass
class WasteSortG1Dex3EnvCfg(ManagerBasedRLEnvCfg):
    # 场景 / 管理器保持和你现在一致
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    rewards = None
    curriculum = None
    commands = None

    def __post_init__(self):
        # 基本步长
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        # —— 下面这些是你对照文件里的 PhysX 细节，建议保留，抓取稳定性更好 —— #
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.contact_offset = 0.01
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 4

        # —— 和对照代码一致：自定义事件管理器（可选，但方便你在运行期手动触发） —— #
        self.event_manager = SimpleEventManager()
        # self.event_manager.register("reset_items_self", SimpleEvent(
        #     func=lambda env: base_mdp.reset_root_state_uniform(
        #         env,
        #         torch.arange(env.num_envs, device=env.device),
        #         pose_range={"x":[0.5,0.9], "y":[-0.3,0.3], "z":[0.05,0.05]},
        #         velocity_range={},
        #         asset_cfg=SceneEntityCfg("carton"),
        #     )
        # ))
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))