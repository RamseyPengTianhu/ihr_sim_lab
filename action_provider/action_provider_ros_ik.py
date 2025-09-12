# action_provider/action_provider_ros_ik.py
import os
import json
import time
import threading
from typing import Optional, Dict, List

import torch

# 你项目里的基类（注意不是 ActionProviderBase）
from action_provider.action_base import ActionProvider

# Unitree G1/H1 低层消息（需要已安装 unitree_ros2 的 unitree_hg 包）
from unitree_hg.msg import LowCmd, MotorCmd

# Pink IK（Isaac Lab 提供的新接口）
from isaaclab.controllers import PinkIKController, PinkIKControllerCfg
from pink.tasks import FrameTask

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


from ros_publisher.g1_hand_publisher import Dex3HandPublisher
from ros_publisher.dex3_hand_state_pub import Dex3HandStatePublisher
from ros_publisher.g1_state_pub import G1StatePublisher



def _build_upper_body_indices(robot_joint_names: List[str]) -> List[int]:
    """根据关节名猜上肢关节索引；若你有显式名单，可通过 --ik_upper_body_names 传入。"""
    keys = [
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
        "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"
    ]
    names = [n.lower() for n in robot_joint_names]
    idx: List[int] = []
    for side in ["left", "right"]:
        for k in keys:
            cand = [i for i, n in enumerate(names) if side in n and k in n]
            if cand:
                idx.append(cand[0])
    # 去重并保持顺序
    seen, uniq = set(), []
    for i in idx:
        if i not in seen:
            uniq.append(i); seen.add(i)
    return uniq


# --------------------------- ROS 节点 ---------------------------

class _ROSIKNode(Node):
    def __init__(self, ee_topic: str, on_ee):
        super().__init__("isaac_ros_ik_bridge")
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._ee_sub = self.create_subscription(PoseStamped, ee_topic, on_ee, qos)


# --------------------------- 主类 ---------------------------

class ROSIKActionProvider(ActionProvider):
    """
    订阅 /ee_target(PoseStamped) → Pink IK 只覆盖上肢关节；
    每步将 IK 结果发布为 unitree_hg/LowCmd 到 /lowcmd（与 G1 实机一致），同时把动作返回给仿真。
    """

    # ====== 构造 / 初始化 ======
    def __init__(self, env, args_cli):
        super().__init__(env)
        self.env = env
        self.args_cli = args_cli
        self.device = env.device

        # 机器人句柄与关节名
        self.robot = env.scene["robot"]
        try:
            self.joint_names: List[str] = list(self.robot.data.joint_names)
        except Exception:
            try:
                n = env.action_manager.num_actions
            except Exception:
                n = int(self.robot.data.joint_pos.shape[-1])
            self.joint_names = [f"joint_{i}" for i in range(n)]

        self.num_joints = len(self.joint_names)

        # 运行参数
        self.dt = float(getattr(env, "physics_dt", 1.0 / 60.0))
        self.max_qdot = torch.tensor(5.0, device=self.device)  # 简单的角速度上限（rad/s）

        # IK / 滤波与发布参数
        self.ee_frame = getattr(args_cli, "ik_ee_frame", "right_hand")
        self.ee_link = getattr(args_cli, "ik_ee_link", "right_hand_palm_link")
        self.keep_last_sec = float(getattr(args_cli, "ik_keep_last_sec", 0.3))
        self.enable_lowpass = bool(getattr(args_cli, "ik_lowpass", True))
        self.lowpass_alpha = float(getattr(args_cli, "ik_lowpass_alpha", 0.2))
        self._lowcmd_topic = getattr(args_cli, "ik_lowcmd_topic", "/lowcmd")
        self.lowcmd_pub_rate = float(getattr(args_cli, "lowcmd_pub_rate", 60.0))  # Hz
        self._last_pub_t = 0.0

        # Debug 开关
        self.ik_debug = bool(getattr(args_cli, "ik_debug", True))
        self.ik_log_every = int(getattr(args_cli, "ik_log_every", 20))
        self._ik_step = 0

        # 与 DDS 对齐的坐标变换模式（frame=world/map/odom 时将目标转换到 root）
        # translate_only: 仅平移（和你之前 DDS 版本一致）；full: 平移+旋转
        self.ik_transform_mode = getattr(args_cli, "ik_transform_mode", "translate_only")

        # 上肢关节索引
        explicit = getattr(args_cli, "ik_upper_body_names", "").strip()
        if explicit:
            name_map = {n.lower(): i for i, n in enumerate(self.joint_names)}
            self.upper_idx = [
                name_map[n.strip().lower()]
                for n in explicit.split(",")
                if n.strip().lower() in name_map
            ]
        else:
            self.upper_idx = _build_upper_body_indices(self.joint_names)

        # -------- Pink IK：FrameTask + Cfg --------
        self.ee_task = FrameTask(
            self.ee_link,          # 末端连杆名（URDF 里的 frame）
            position_cost=1.0,
            orientation_cost=1.0,
        )

        # 可通过 CLI 覆盖默认 URDF 路径
        urdf_path = getattr(
            args_cli,
            "ik_urdf_path",
            "/home/tianhup/Desktop/xr_teleoperate/assets/g1/g1_body29_hand14.urdf",
        )
        mesh_path = getattr(
            args_cli,
            "ik_mesh_path",
            "/home/tianhup/Desktop/xr_teleoperate/assets/g1",
        )

        cfg = PinkIKControllerCfg()
        cfg.urdf_path = urdf_path
        cfg.mesh_path = mesh_path
        cfg.joint_names = list(self.joint_names)
        cfg.variable_input_tasks = [self.ee_task]   # 运行时会改这个任务的 target
        cfg.fixed_input_tasks = []
        cfg.show_ik_warnings = True

        self.ik = PinkIKController(cfg, str(self.device))

        # 目标缓存
        self._target: Optional[Dict[str, torch.Tensor]] = None
        self._target_t: float = 0.0
        self._last_ub_q: Optional[torch.Tensor] = None

        # 初始关节、限位（统一成一维）
        try:
            q0 = torch.as_tensor(self.robot.data.joint_pos, device=self.device, dtype=torch.float32)
        except Exception:
            q0 = torch.zeros((self.num_joints,), device=self.device, dtype=torch.float32)
        if q0.ndim == 2 and q0.shape[0] == 1:
            q0 = q0[0]
        self.last_q = q0.clone()

        try:
            jl = torch.as_tensor(self.robot.data.joint_limits[0], device=self.device, dtype=torch.float32)
            ju = torch.as_tensor(self.robot.data.joint_limits[1], device=self.device, dtype=torch.float32)
        except Exception:
            jl = torch.full((self.num_joints,), -10.0, device=self.device)
            ju = torch.full((self.num_joints,), +10.0, device=self.device)
        if jl.ndim == 2 and jl.shape[0] == 1:
            jl = jl[0]
        if ju.ndim == 2 and ju.shape[0] == 1:
            ju = ju[0]
        self.joint_lower = jl
        self.joint_upper = ju

        # 仿真关节 → G1 35路电机映射
        self.sim_to_g1 = None
        try:
            cfg_path = os.path.join(
                os.path.dirname(__file__), "..", "ros_publisher", "config", "g1_joint_mapping.json"
            )
            cfg_path = os.path.normpath(cfg_path)
            with open(cfg_path, "r") as f:
                self.sim_to_g1 = json.load(f).get("sim_to_g1")
            if not self.sim_to_g1:
                print("[ROSIK] sim_to_g1 empty, fallback to default mapping.")
                self.sim_to_g1 = None
        except Exception as e:
            print(f"[ROSIK] load sim_to_g1 failed: {e}")
            self.sim_to_g1 = None  # fallback: 在 _publish_lowcmd 里默认顺序处理

        # 一次性打印信息
        try:
            print("[ROSIK][DEBUG] num_joints(Isaac) =", len(self.joint_names))
            for i, n in enumerate(self.joint_names[:]):  # 防刷屏
                print(f"[ROSIK][DEBUG] sim_joint[{i:02d}] = {n}")
            print("[ROSIK][DEBUG] upper_idx =", self.upper_idx, " (len =", len(self.upper_idx), ")")
            print("[ROSIK][DEBUG] sim_to_g1 (raw) =", self.sim_to_g1 if self.sim_to_g1 is not None else "DEFAULT [0..min(num_sim,35)-1]")
        except Exception as e:
            print("[ROSIK][DEBUG] mapping print failed:", e)

        # ROS2 / 线程
        self._ros_ctx_inited = False
        self._ros_node: Optional[_ROSIKNode] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._lowcmd_pub = None

        # 初始化 ROS
        self._init_ros(args_cli)


        # ====== 加：LowState 发布器 ======
        self._g1_state_pub = G1StatePublisher(
            node_name="g1_state_publisher",
            sim_to_g1=self.sim_to_g1,
            rate_hz=60.0,
            joint_names=self.joint_names,   # 可选，用来在 JointState 里带名字
        )
        # 把 state publisher 挂到 ROS executor
        if self._executor:
            self._executor.add_node(self._g1_state_pub)

        dex3_map_json = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "ros_publisher", "config", "dex3_joint_mapping.json")
        )
        self._dex3_pub = Dex3HandPublisher(
            node=self._ros_node,
            which=getattr(args_cli, "dex3_side", "right"),   # "right"/"left"
            right_topic=getattr(args_cli, "dex3_right_topic", "/dex3/right/cmd"),
            left_topic=getattr(args_cli, "dex3_left_topic", "/dex3/left/cmd"),
            mapping_json_path=dex3_map_json,
            sim_to_dex3=getattr(args_cli, "sim_to_dex3", None), # 也可从 CLI 传 list
        )

        self._dex3_state_pub_r = Dex3HandStatePublisher(
            node=self._ros_node,
            which="right",
            # 映射请根据你的 joint 打印填（示例）：
            # sim_to_dex3=[32, 33, 34, 38, 39, 40, 41],  # 右手 index0/middle0/thumb0/index1/middle1/thumb1/???（根据你的实际命名改！）
            mapping_json_path=dex3_map_json,
            rate_hz=30.0,
        )

    # ====== ROS 初始化 / 清理 ======
    def _init_ros(self, args_cli):
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_ctx_inited = True

        ee_topic = getattr(args_cli, "ros_ee_topic", "/ee_target")

        def on_ee(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            pos_t = torch.tensor([p.x, p.y, p.z], device=self.device, dtype=torch.float32)
            # ROS 默认 xyzw → 转 wxyz
            quat_xyzw = torch.tensor([q.x, q.y, q.z, q.w], device=self.device, dtype=torch.float32)
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            # 单位化保护
            n = torch.linalg.norm(quat_wxyz)
            if torch.abs(n - 1.0) > 1e-3:
                quat_wxyz = quat_wxyz / (n + 1e-8)
            frame_in = (msg.header.frame_id or "world").lower()

            # Debug: 原始输入
            if self.ik_debug:
                print(f"[ROSIK][EE] recv frame='{frame_in}' pos={pos_t.tolist()} quat_wxyz={quat_wxyz.tolist()}")

            # 与 DDS 对齐：world/map/odom → root（默认只平移）
            if frame_in in ("world", "map", "odom"):
                pos_local, quat_local = self._world_to_root(pos_t, quat_wxyz, mode=self.ik_transform_mode)
                frame_out = "root"
            else:
                # base/base_link/root 都视为 root
                pos_local, quat_local = pos_t, quat_wxyz
                frame_out = frame_in

            if self.ik_debug:
                print(f"[ROSIK][EE] use frame='{frame_out}' pos_local={pos_local.tolist()} quat_local_wxyz={quat_local.tolist()}")

            self._target = {"pos": pos_local, "quat": quat_local, "frame": frame_out}
            self._target_t = time.time()

        # 创建 node（带回调）
        self._ros_node = _ROSIKNode(ee_topic, on_ee)

        # /lowcmd 发布器（RELIABLE）
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self._lowcmd_pub = self._ros_node.create_publisher(LowCmd, self._lowcmd_topic, qos_rel)

        # 多线程 executor
        exec_ = MultiThreadedExecutor()
        exec_.add_node(self._ros_node)

        def spin():
            try:
                exec_.spin()
            finally:
                exec_.shutdown()

        self._executor = exec_
        self._spin_thread = threading.Thread(target=spin, daemon=True)
        self._spin_thread.start()

    def cleanup(self):
        if not self._ros_ctx_inited:
            return
        try:
            if self._executor is not None:
                self._executor.shutdown()
        except Exception:
            pass
        try:
            if self._ros_node is not None:
                self._ros_node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            if self._spin_thread is not None and self._spin_thread.is_alive():
                self._spin_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self._g1_state_pub is not None:
                self._g1_state_pub.destroy_node()
        except Exception:
            pass

    # ====== 坐标系辅助 ======
    def _world_to_root(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        # 先用平移简化；若抓取时姿态不准再补旋转
        root_state = self.robot.data.root_state_w  # [N,13]
        root_pos = torch.tensor(root_state[0, :3], device=self.device, dtype=torch.float32)
        pos_local = pos_w - root_pos
        return pos_local, quat_w

        

    # ====== /lowcmd 发布 ======
    def _publish_lowcmd(self, full_q: torch.Tensor):
        """将仿真关节位置 full_q（[1, num_sim]）映射并发布为 unitree_hg/LowCmd。"""
        if self._lowcmd_pub is None:
            return

        # 限频
        if self.lowcmd_pub_rate > 0.0:
            now = time.time()
            if (now - self._last_pub_t) < (1.0 / self.lowcmd_pub_rate):
                return
            self._last_pub_t = now

        # 数值保护
        if not torch.isfinite(full_q).all():
            full_q = torch.nan_to_num(full_q, nan=0.0, posinf=0.0, neginf=0.0)

        q = full_q[0].detach().cpu().tolist()
        N_SIM = len(q)
        N_G1 = 35  # G1 固定 35 路

        # 映射：若无配置，则默认 0..min(N_SIM, 35)-1
        sim_to_g1 = self.sim_to_g1 or list(range(min(N_SIM, N_G1)))

        # Debug：映射预览
        if self.ik_debug and (self._ik_step % self.ik_log_every == 0):
            preview = []
            for i_sim in range(min(N_SIM, len(sim_to_g1), 16)):  # 打印前 16 个映射
                i_g1 = sim_to_g1[i_sim]
                if 0 <= i_g1 < N_G1:
                    preview.append((i_sim, i_g1, float(q[i_sim])))
            print(f"[ROSIK][MAP] sim_idx->g1_idx:q = {preview}")

        msg = LowCmd()
        msg.mode_pr = 0
        msg.mode_machine = 0

        # 先 35 个占位
        msg.motor_cmd = [MotorCmd() for _ in range(N_G1)]
        for i in range(N_G1):
            mc = msg.motor_cmd[i]
            mc.mode = 0x00
            mc.q = 0.0
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = 0.0
            mc.reserve = 0

        # 写入映射
        for i_sim, qval in enumerate(q):
            if i_sim >= len(sim_to_g1):
                break
            i_g1 = sim_to_g1[i_sim]
            if 0 <= i_g1 < N_G1:
                mc = msg.motor_cmd[i_g1]
                mc.mode = 0x01
                mc.q = float(qval)

        self._lowcmd_pub.publish(msg)

    # ====== 主接口：给控制器的动作 ======
    @torch.no_grad()
    def get_action(self, obs=None, t=None) -> torch.Tensor:
        """返回给环境的动作，同时发布 /lowcmd。"""
        self._ik_step += 1
        full_action = torch.zeros((1, self.num_joints), device=self.device, dtype=torch.float32)

        # 目标是否新鲜
        fresh = (self._target is not None) and ((time.time() - self._target_t) <= self.keep_last_sec)

        if fresh and len(self.upper_idx) > 0:
            # 读取目标（root系；如果接收是 world 已转过）
            pos_t: torch.Tensor = self._target["pos"]          # (3,)
            quat_wxyz: torch.Tensor = self._target["quat"]     # (4,)
            frame = self._target.get("frame", "world")


            # 组装 Pink 的目标（保持与 DDS 版一致，直接用当前 Transform 的 copy 再改平移/旋转）

            T_curr = self.ik.pink_configuration.get_transform_frame_to_world(self.ee_link)
            T_tgt = T_curr.copy()
            T_tgt.translation = pos_t.detach().cpu().numpy().astype("float64")
            R = self._quat_wxyz_to_rot(quat_wxyz).detach().cpu().numpy().astype("float64")
            T_tgt.rotation = R
            print('self.ee_link:',self.ee_link)

            print('T_tgt:',T_tgt)

            self.ee_task.set_target(T_tgt)

            # IK 输入
            q_prev_np = self.last_q.detach().cpu().numpy().astype("float64")
            if q_prev_np.ndim == 2 and q_prev_np.shape[0] == 1:
                q_prev_np = q_prev_np[0]

            if self.ik_debug and (self._ik_step % self.ik_log_every == 0):
                upreview = [(i, float(self.last_q[i].item())) for i in self.upper_idx[:14]]
                print(f"[ROSIK][IK-IN] step={self._ik_step} pos={pos_t.tolist()} quat_wxyz={quat_wxyz.tolist()} q_prev(upper head)={upreview}")

            # 计算
            q_des_np = self.ik.compute(q_prev_np, float(self.dt))

            # 尺寸规整
            if isinstance(q_des_np, (list, tuple)):
                import numpy as np
                q_des_np = np.array(q_des_np, dtype="float64")
            if getattr(q_des_np, "ndim", 1) == 2 and q_des_np.shape[0] == 1:
                q_des_np = q_des_np[0]

            q_des = torch.as_tensor(q_des_np, device=self.device, dtype=torch.float32)

            # 限位 + 限速
            q_raw = q_des.clone()
            q_des = torch.clamp(q_des, self.joint_lower, self.joint_upper)
            q_des = self._apply_rate_limit(q_des, self.last_q, float(self.dt))

            # 低通滤波（只对上肢段）
            if self.enable_lowpass and (self._last_ub_q is not None):
                a = float(self.lowpass_alpha)
                ub_q = q_des[self.upper_idx]
                if self._last_ub_q.shape == ub_q.shape:
                    ub_q = (1 - a) * self._last_ub_q + a * ub_q
                q_des = q_des.clone()
                q_des[self.upper_idx] = ub_q
                self._last_ub_q = ub_q.detach().clone()
            else:
                self._last_ub_q = q_des[self.upper_idx].detach().clone()

            # 只把上肢写进动作（其他关节保持 0）
            # full_action[0, self.upper_idx] = q_des[self.upper_idx]
            full_action[0, :] = q_des


            # 记录 last_q（全量）
            self.last_q = q_des.detach().clone()

            if self.ik_debug and (self._ik_step % self.ik_log_every == 0):
                up_out = [(i, float(q_raw[i].item()), float(q_des[i].item())) for i in self.upper_idx[:14]]
                print(f"[ROSIK][IK-OUT] step={self._ik_step} upper(raw->clamped/rate/LP head)={up_out}")

        elif (self._last_ub_q is not None) and (len(self.upper_idx) == self._last_ub_q.shape[0]):
            # 没新目标则保持上一姿态
            full_action[0, self.upper_idx] = self._last_ub_q

        # 发布到 /lowcmd
        self._publish_lowcmd(full_action)

        # ====== 加：发布 /lowstate ======
        if self._g1_state_pub:
            self._g1_state_pub.publish(self.env)

        try:
            full_action = self._dex3_pub.publish_from_full_action(
            full_action,
            inject_cached_q7=True,     # 把订阅/缓存的 q7 注入 full_action
            return_updated=True        # 返回覆盖后的 full_action
        )
        except Exception as e:
            print(f"[Dex3] publish failed: {e}")

        if hasattr(self, "_dex3_state_pub_r"):
            self._dex3_state_pub_r.publish(self.env)

        return full_action

    # ====== 实用函数 ======
    @staticmethod
    def _quat_wxyz_to_rot(qwxyz: torch.Tensor) -> torch.Tensor:
        """四元数(w,x,y,z) → 旋转矩阵 3x3（以 torch 表达）"""
        w, x, y, z = qwxyz
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        return torch.tensor([
            [ww + xx - yy - zz, 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     ww - xx + yy - zz, 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     ww - xx - yy + zz]
        ], device=qwxyz.device, dtype=torch.float32)

    @staticmethod
    def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
        return q / (q.norm(p=2) + 1e-8)

    def _apply_rate_limit(self, q_des: torch.Tensor, q_prev: torch.Tensor, dt: float):
        dq = q_des - q_prev
        dq = torch.clamp(dq, -self.max_qdot * dt, self.max_qdot * dt)
        return q_prev + dq