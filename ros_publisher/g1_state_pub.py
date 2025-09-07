# ros_publisher/g1_state_pub.py
import time
import math
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import defaultdict

from unitree_hg.msg import LowState, MotorState, LowCmd
from sensor_msgs.msg import JointState


class G1StatePublisher(Node):
    """仿真专用：把 Isaac Lab 的关节/IMU 状态封装成 G1 的 /lowstate。
    - 固定发布 35 路 MotorState（未映射的通道置零）
    - 与真机同时跑时请不要启用，避免 /lowstate 冲突
    """
    def __init__(
        self,
        node_name: str = "g1_state_publisher",
        sim_to_g1: Optional[List[int]] = None,
        rate_hz: float = 60.0,
        joint_names: Optional[List[str]] = None,   # <—— 传进来，便于 debug
    ):
        super().__init__(node_name)

        # QoS
        qos_be = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,   # 状态流一般 BEST_EFFORT 即可
            history=HistoryPolicy.KEEP_LAST
        )
        qos_rel = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # 话题
        self.pub_low = self.create_publisher(LowState, "/lowstate", qos_be)
        self.pub_debug = self.create_publisher(JointState, "/g1/lowstate_debug", qos_be)

        # —— 关键：镜像 /lowcmd 的 mode —— #
        self._last_lowcmd_mode = defaultdict(int)   # g1_idx -> mode
        self.create_subscription(LowCmd, "/lowcmd", self._on_lowcmd, qos_rel)

        # 配置
        self.N_G1 = 35
        self.sim_to_g1 = sim_to_g1
        self.rate_hz = float(rate_hz)
        self.joint_names = joint_names or []

        # 状态
        self._last_pub_t = 0.0
        self._tick = 0

    # === 订阅 /lowcmd：把每个电机的 mode 记下来（用于 state 镜像） ===
    def _on_lowcmd(self, msg: LowCmd):
        for i, mc in enumerate(msg.motor_cmd):
            self._last_lowcmd_mode[i] = int(mc.mode)

    # ---------------- 内部小工具 ----------------
    @staticmethod
    def _safe_float(x) -> float:
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return 0.0

    @staticmethod
    def _quat_xyzw_to_wxyz(q):
        # 传进来可能是 torch/tensor/numpy：统一取值
        try:
            x, y, z, w = [float(q[i]) for i in range(4)]
        except Exception:
            return [1.0, 0.0, 0.0, 0.0]
        return [w, x, y, z]

    @staticmethod
    def _quat_to_rpy_wxyz(qwxyz):
        # 简单从四元数近似解算 rpy（用于 LowState.imu_state.rpy）
        w, x, y, z = qwxyz
        # roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return [roll, pitch, yaw]

    # --------------- 对外发布接口：在主循环里调用 ----------------
    def publish(self, env) -> None:
        # 限频
        if self.rate_hz > 0:
            now = time.time()
            if (now - self._last_pub_t) < (1.0 / self.rate_hz):
                return
            self._last_pub_t = now

        msg = LowState()

        # 基本字段
        msg.version = [1, 0]        # 自定：主/次版本
        msg.mode_pr = 0
        msg.mode_machine = 0
        self._tick = (self._tick + 1) & 0xFFFFFFFF
        msg.tick = self._tick

        # 读取仿真
        try:
            robot = env.scene["robot"]
            q  = robot.data.joint_pos[0]      # [num_sim]
            dq = robot.data.joint_vel[0]      # [num_sim]
            num_sim = int(q.shape[0])

            # root状态：用于 IMU 粗略填充
            root = robot.data.root_state_w[0]  # [13] xyz + quat(xyzw) + linvel + angvel
            root_quat_xyzw = root[3:7]
            root_angvel = root[10:13]
        except Exception as e:
            self.get_logger().warn(f"read sim state failed: {e}")
            return

        # IMU 近似：用 base/root 代表（仅为兼容接口，数值不必与真机一致）
        qwxyz = self._quat_xyzw_to_wxyz(root_quat_xyzw)
        msg.imu_state.quaternion  = [float(v) for v in qwxyz]              # wxyz
        msg.imu_state.gyroscope   = [self._safe_float(v) for v in root_angvel]  # rad/s
        msg.imu_state.accelerometer = [0.0, 0.0, 9.81]                     # 简单给重力
        msg.imu_state.rpy = [float(v) for v in self._quat_to_rpy_wxyz(qwxyz)]
        msg.imu_state.temperature = 25  # 假值

        # 电机状态（固定 35 路）
        sim_to_g1 = self.sim_to_g1 or list(range(min(num_sim, self.N_G1)))
        msg.motor_state = [MotorState() for _ in range(self.N_G1)]
        # 先清零 + 镜像 mode（如果有 /lowcmd）
        for i in range(self.N_G1):
            ms = msg.motor_state[i]
            ms.mode = int(self._last_lowcmd_mode.get(i, 0))  # <—— 镜像 /lowcmd
            ms.q = 0.0
            ms.dq = 0.0
            ms.ddq = 0.0
            ms.tau_est = 0.0
            ms.temperature = [25, 25]
            ms.vol = 0.0
            ms.sensor = [0, 0]
            ms.motorstate = 0

        # 写入映射到的通道
        for i_sim in range(min(num_sim, len(sim_to_g1))):
            i_g1 = sim_to_g1[i_sim]
            if 0 <= i_g1 < self.N_G1:
                ms = msg.motor_state[i_g1]
                ms.q  = self._safe_float(q[i_sim])
                ms.dq = self._safe_float(dq[i_sim])
                # ms.mode 保持镜像值（如果你希望“被映射就强制=1”，改成：ms.mode = max(ms.mode, 1)）

        # 其他保留字段
        msg.wireless_remote = [0]*40
        msg.reserve = [0]*4
        msg.crc = 0

        self.pub_low.publish(msg)

        # 伴随 JointState 调试（带名字）
        try:
            dbg = JointState()
            dbg.header.stamp = self.get_clock().now().to_msg()
            names, pos, vel = [], [], []
            for i_sim in range(min(num_sim, len(sim_to_g1))):
                i_g1 = sim_to_g1[i_sim]
                nm = (self.joint_names[i_sim] if i_sim < len(self.joint_names) else f"joint_{i_sim}") + f" [g1:{i_g1}]"
                names.append(nm)
                pos.append(self._safe_float(q[i_sim]))
                vel.append(self._safe_float(dq[i_sim]))
            dbg.name = names
            dbg.position = pos
            dbg.velocity = vel
            self.pub_debug.publish(dbg)
        except Exception as e:
            self.get_logger().warn(f"debug JointState publish failed: {e}")
