# action_provider/action_provider_ros.py
import os
import json
import time
import threading
from typing import Optional, List

import torch
from action_provider.action_base import ActionProvider

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Unitree 消息
from unitree_hg.msg import LowCmd

# 发布器
from ros_publisher.g1_state_pub import G1StatePublisher
from ros_publisher.g1_hand_publisher import Dex3HandPublisher
from ros_publisher.dex3_hand_state_pub import Dex3HandStatePublisher


# -------------------- 工具 --------------------
def _ensure_torch_1d(x, like_tensor=None):
    dev = getattr(like_tensor, "device", None)
    if isinstance(x, torch.Tensor):
        t = x.to(device=dev, dtype=torch.float32)
    else:
        t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    return t.reshape(-1)


# -------------------- 仅订阅 /lowcmd 的 ROS 节点 --------------------
class _ROSLowCmdNode(Node):
    def __init__(self, topic_name: str, on_lowcmd):
        super().__init__("isaac_ros_lowcmd_bridge")
        qos_rel = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,   # 与发布端一致
            history=HistoryPolicy.KEEP_LAST,
        )
        self._sub = self.create_subscription(LowCmd, topic_name, on_lowcmd, qos_rel)


# -------------------- 主类：/lowcmd -> env action --------------------
class ROSLowCmdActionProvider(ActionProvider):
    """
    订阅 /lowcmd（unitree_hg/LowCmd），把 q 映射成环境动作返回。
    不直接写 robot 的 set_*，避免被控制栈覆盖。
    同时发布 /lowstate 与 Dex3 状态。
    """

    def __init__(self, env, args_cli):
        super().__init__(env)
        self.env = env
        self.args_cli = args_cli
        self.device = env.device

        # 可配置：是否仅在 mode==1 时应用（默认 True）
        self.require_mode1: bool = bool(getattr(args_cli, "require_mode1", True))

        # 机器人 & 关节
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

        # ===== 加载映射（自动识别 sim_to_g1 / g1_to_sim） =====
        self.N_G1 = 35
        self.sim_to_g1 = None
        self._load_and_prepare_mapping()

        print("[ROSLowCmd] num_joints =", self.num_joints)
        print("[ROSLowCmd] sim_to_g1 =", self.sim_to_g1 if self.sim_to_g1 else "DEFAULT")

        # ROS init
        self._ros_ctx_inited = False
        self._ros_node: Optional[_ROSLowCmdNode] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None

        self._last_lowcmd: Optional[LowCmd] = None
        self._last_lowcmd_ts: float = 0.0
        self._lowcmd_recv_cnt: int = 0
        self._lowcmd_lock = threading.Lock()
        self._init_ros(args_cli)

        # 发布 /lowstate
        self._g1_state_pub = G1StatePublisher(
            node_name="g1_state_publisher",
            sim_to_g1=self.sim_to_g1,
            rate_hz=60.0,
            joint_names=self.joint_names,
        )
        if self._executor:
            self._executor.add_node(self._g1_state_pub)

        # Dex3（可用可不用）
        dex3_map_json = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "ros_publisher", "config", "dex3_joint_mapping.json")
        )
        self._dex3_pub = Dex3HandPublisher(
            node=self._ros_node,
            which=getattr(args_cli, "dex3_side", "right"),
            right_topic=getattr(args_cli, "dex3_right_topic", "/dex3/right/cmd"),
            left_topic=getattr(args_cli, "dex3_left_topic", "/dex3/left/cmd"),
            mapping_json_path=dex3_map_json,
            sim_to_dex3=getattr(args_cli, "sim_to_dex3", None),
        )
        self._dex3_state_pub_r = Dex3HandStatePublisher(
            node=self._ros_node, which="right", mapping_json_path=dex3_map_json, rate_hz=30.0
        )

        # 记录上一上肢姿态（无指令时保持）
        self.upper_idx = getattr(self, "upper_idx", [])  # 若外部已提供则复用
        self._last_ub_q: Optional[torch.Tensor] = None
        self.last_q: Optional[torch.Tensor] = None
        self.dt = getattr(self, "dt", 1 / 60.0)
        self._ik_step = 0

    # -------------------- 映射加载与自检 --------------------
    def _load_and_prepare_mapping(self):
        # 获取仿真关节数
        try:
            n_sim = int(self.robot.data.joint_pos.shape[-1])
        except Exception:
            n_sim = self.num_joints

        # 1) 先尝试读 JSON（sim_to_g1 或 g1_to_sim）
        def _load_mapping_json():
            cfg_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "ros_publisher", "config", "g1_joint_mapping.json")
            )
            try:
                with open(cfg_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ROSLowCmd] mapping json load failed: {e}")
                return {}

        cfg = _load_mapping_json()

        def _invert_g1_to_sim(g1_to_sim, n_sim, N_G1=35):
            sim_to_g1 = [-1] * n_sim
            for g1_idx, sim_idx in enumerate(g1_to_sim):
                try:
                    if 0 <= sim_idx < n_sim:
                        sim_to_g1[sim_idx] = int(g1_idx)
                except Exception:
                    pass
            return sim_to_g1

        sim_to_g1 = None
        if isinstance(cfg.get("sim_to_g1"), list) and len(cfg["sim_to_g1"]) > 0:
            sim_to_g1 = cfg["sim_to_g1"]
        elif isinstance(cfg.get("g1_to_sim"), list) and len(cfg["g1_to_sim"]) > 0:
            sim_to_g1 = _invert_g1_to_sim(cfg["g1_to_sim"], n_sim, 35)
            print("[ROSLowCmd] mapping provided as g1_to_sim; auto-inverted to sim_to_g1.")

        # 2) 若无 JSON，则按“真机枚举顺序 + 关节名”自动构建
        if sim_to_g1 is None:
            print("[ROSLowCmd] mapping missing; building NAME-BASED mapping by G1 enum order.")

            # —— G1 真机枚举（你提供的顺序）——
            G1_ENUM = {
                # Legs - Left
                "LeftHipPitch": 0, "LeftHipRoll": 1, "LeftHipYaw": 2, "LeftKnee": 3,
                "LeftAnklePitch": 4, "LeftAnkleRoll": 5,  # A/B 别名同位
                # Legs - Right
                "RightHipPitch": 6, "RightHipRoll": 7, "RightHipYaw": 8, "RightKnee": 9,
                "RightAnklePitch": 10, "RightAnkleRoll": 11,
                # Waist
                "WaistYaw": 12, "WaistRoll": 13, "WaistPitch": 14,  # 某些机型可能锁止，但仍给索引
                # Arms - Left
                "LeftShoulderPitch": 15, "LeftShoulderRoll": 16, "LeftShoulderYaw": 17, "LeftElbow": 18,
                "LeftWristRoll": 19, "LeftWristPitch": 20, "LeftWristYaw": 21,
                # Arms - Right
                "RightShoulderPitch": 22, "RightShoulderRoll": 23, "RightShoulderYaw": 24, "RightElbow": 25,
                "RightWristRoll": 26, "RightWristPitch": 27, "RightWristYaw": 28,
            }

            # —— 关节名字到 G1 名称的模糊表 —— #
            # 说明：key 是小写匹配关键词的元组（任一匹配即命中），value 是上面的 G1_ENUM 名称
            PATTERNS = {
                # Legs - Left
                ("left_hip_pitch",): "LeftHipPitch",
                ("left_hip_roll",):  "LeftHipRoll",
                ("left_hip_yaw",):   "LeftHipYaw",
                ("left_knee",):      "LeftKnee",
                ("left_ankle_pitch", "left_ankle_b"): "LeftAnklePitch",  # 别名 B
                ("left_ankle_roll",  "left_ankle_a"): "LeftAnkleRoll",   # 别名 A
                # Legs - Right
                ("right_hip_pitch",): "RightHipPitch",
                ("right_hip_roll",):  "RightHipRoll",
                ("right_hip_yaw",):   "RightHipYaw",
                ("right_knee",):      "RightKnee",
                ("right_ankle_pitch", "right_ankle_b"): "RightAnklePitch",
                ("right_ankle_roll",  "right_ankle_a"): "RightAnkleRoll",
                # Waist
                ("waist_yaw",):   "WaistYaw",
                ("waist_roll",):  "WaistRoll",
                ("waist_pitch",): "WaistPitch",
                # Arms - Left
                ("left_shoulder_pitch",): "LeftShoulderPitch",
                ("left_shoulder_roll",):  "LeftShoulderRoll",
                ("left_shoulder_yaw",):   "LeftShoulderYaw",
                ("left_elbow",):          "LeftElbow",
                ("left_wrist_roll", "left_wist_roll"):   "LeftWristRoll",   # 兼容 wist 拼写
                ("left_wrist_pitch","left_wist_pitch"):  "LeftWristPitch",
                ("left_wrist_yaw",  "left_wist_yaw"):    "LeftWristYaw",
                # Arms - Right
                ("right_shoulder_pitch",): "RightShoulderPitch",
                ("right_shoulder_roll",):  "RightShoulderRoll",
                ("right_shoulder_yaw",):   "RightShoulderYaw",
                ("right_elbow",):          "RightElbow",
                ("right_wrist_roll","right_wist_roll"):  "RightWristRoll",
                ("right_wrist_pitch","right_wist_pitch"):"RightWristPitch",
                ("right_wrist_yaw",  "right_wist_yaw"):  "RightWristYaw",
            }

            # 小写名表
            lower_names = [str(nm).lower() for nm in self.joint_names]

            # 构造 sim_to_g1，默认 -1（不映射）
            sim_to_g1 = [-1] * n_sim

            # 对每个仿真关节名，尝试匹配到一个 G1 关节
            for i_sim, nm_low in enumerate(lower_names):
                matched_g1_name = None
                for keys, g1_name in PATTERNS.items():
                    if any(k in nm_low for k in keys):
                        matched_g1_name = g1_name
                        break
                if matched_g1_name is None:
                    continue
                g1_idx = G1_ENUM[matched_g1_name]
                # 若该 G1 索引已被别的关节占用，跳过（避免一对多）
                if g1_idx in sim_to_g1:
                    continue
                sim_to_g1[i_sim] = g1_idx

            # 额外提示：若必须保证 0..28 都有对应，可在此处打印缺失的 g1 索引
            missing = sorted(set(G1_ENUM.values()) - {idx for idx in sim_to_g1 if idx >= 0})
            if missing:
                print(f"[ROSLowCmd][WARN] unmapped G1 indices: {missing} (OK if你的仿真里没有这些关节/被锁)")

        else:
            # 直接使用/修正来自 JSON 的 sim_to_g1
            sim_to_g1 = list(sim_to_g1)[:n_sim]
            sim_to_g1 = [int(idx) if isinstance(idx, int) and 0 <= idx < 35 else -1 for idx in sim_to_g1]

        self.sim_to_g1 = sim_to_g1

        # 打印若干映射用于人工核对
        preview = []
        head = min(40, len(self.sim_to_g1))
        for i_sim in range(head):
            i_g1 = self.sim_to_g1[i_sim]
            nm = self.joint_names[i_sim] if i_sim < len(self.joint_names) else f"joint_{i_sim}"
            preview.append(f"{i_sim}:{nm} -> g1:{i_g1}")
        print("[ROSLowCmd] mapping preview:", "; ".join(preview))


    # -------------------- ROS 生命周期 --------------------
    def _init_ros(self, args_cli):
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_ctx_inited = True

        lowcmd_topic = getattr(args_cli, "ik_lowcmd_topic", "/lowcmd")

        def on_lowcmd(msg: LowCmd):
            with self._lowcmd_lock:
                self._last_lowcmd = msg
                self._last_lowcmd_ts = time.time()
                self._lowcmd_recv_cnt += 1

        self._ros_node = _ROSLowCmdNode(lowcmd_topic, on_lowcmd)
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

    # -------------------- 主接口：返回 env 动作 --------------------
    @torch.no_grad()
    def get_action(self, obs=None, t=None) -> torch.Tensor:
        self._ik_step += 1
        full_action = torch.zeros((1, self.num_joints), device=self.device, dtype=torch.float32)

        # 当前姿态
        try:
            q_cur = self.robot.data.joint_pos[0]
            n = int(q_cur.shape[0])
        except Exception:
            # 仍发布状态，返回 0
            if self._g1_state_pub:
                self._g1_state_pub.publish(self.env)
            if hasattr(self, "_dex3_state_pub_r") and (self._dex3_state_pub_r is not None):
                self._dex3_state_pub_r.publish(self.env)
            return full_action

        # 取一份稳定的 /lowcmd
        with self._lowcmd_lock:
            msg = self._last_lowcmd

        # 没收到指令：保持上一上肢
        if (msg is None) or (not hasattr(msg, "motor_cmd")) or (len(msg.motor_cmd) == 0):
            if (self._last_ub_q is not None) and (len(self.upper_idx) == self._last_ub_q.shape[0]):
                full_action[0, self.upper_idx] = self._last_ub_q
            if self._g1_state_pub:
                self._g1_state_pub.publish(self.env)
            if hasattr(self, "_dex3_state_pub_r") and (self._dex3_state_pub_r is not None):
                self._dex3_state_pub_r.publish(self.env)
            return full_action

        # 解析 /lowcmd → q_des
        sim_to_g1 = self.sim_to_g1 or list(range(min(self.N_G1, n)))
        q_des = q_cur.clone()
        applied = 0
        for i_sim, i_g1 in enumerate(sim_to_g1):
            if i_g1 is None or i_g1 < 0:   # 跳过未映射
                continue
            if i_g1 >= len(msg.motor_cmd):
                continue
            mc = msg.motor_cmd[i_g1]
            if self.require_mode1 and int(getattr(mc, "mode", 0)) != 1:
                continue
            q_des[i_sim] = float(getattr(mc, "q", 0.0))
            applied += 1

        # 限位
        try:
            limits = self.robot.data.joint_pos_limits[0]  # [n,2]
            joint_lower, joint_upper = limits[:, 0], limits[:, 1]
            q_des = torch.clamp(q_des, joint_lower, joint_upper)
        except Exception:
            pass

        # 限速（如你已有该函数）
        if hasattr(self, "_apply_rate_limit"):
            q_des = self._apply_rate_limit(q_des, getattr(self, "last_q", q_cur), float(self.dt))

        # 记录 & 返回
        full_action[0, :] = q_des
        self.last_q = q_des.detach().clone()
        if len(self.upper_idx) > 0:
            self._last_ub_q = q_des[self.upper_idx].detach().clone()

        # 发布状态
        if self._g1_state_pub:
            self._g1_state_pub.publish(self.env)
        try:
            full_action = self._dex3_pub.publish_from_full_action(
                full_action, inject_cached_q7=True, return_updated=True
            )
        except Exception:
            pass
        if hasattr(self, "_dex3_state_pub_r") and (self._dex3_state_pub_r is not None):
            self._dex3_state_pub_r.publish(self.env)

        # 调试节流
        if (self._ik_step % 60 == 0):
            print(f"[LC→ACT] applied joints: {applied}, recv_cnt: {self._lowcmd_recv_cnt}, require_mode1={self.require_mode1}")

        return full_action
