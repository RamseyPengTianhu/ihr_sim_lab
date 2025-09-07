# ros_publisher/dex3_hand_state_pub.py
import time
import json
import os
from typing import List, Optional

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from unitree_hg.msg import HandState, MotorState, IMUState, PressSensorState


class Dex3HandStatePublisher:
    """
    将仿真的手部关节状态发布为 unitree_hg/HandState：
      - 话题：/dex3/right/state 或 /dex3/left/state（取决于 which）
      - 电机数：默认 7 路（Dex3）
      - 压力传感器：仿真里通常没有 → 以 7 个空/零默认填充
      - IMU：仿真一般也没有 → 用 root/基座粗略近似，或默认值
      - 电源/错误码：默认值

    使用：
      pub = Dex3HandStatePublisher(node=_ros_node, which="right",
                                   sim_to_dex3=[32,33,34,38,39,??,??],  # 你自己的映射
                                   rate_hz=60.0)
      ...
      pub.publish(env)  # 在主循环中按需调用（内部自带限频）
    """

    def __init__(
        self,
        node,
        which: str = "right",                           # "right" 或 "left"
        right_topic: str = "/dex3/right/state",
        left_topic: str = "/dex3/left/state",
        sim_to_dex3: Optional[List[int]] = None,        # 仿真关节索引 → 7 路 dex3 顺序
        mapping_json_path: Optional[str] = None,        # 可选 JSON：{"right":[...], "left":[...]}
        rate_hz: float = 60.0,
        num_motors: int = 7,
    ):
        self._node = node
        self.which = which.lower()
        self.num_motors = int(num_motors)
        self.rate_hz = float(rate_hz)
        self._last_pub_t = 0.0

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,   # 状态流 BEST_EFFORT 足够
            history=HistoryPolicy.KEEP_LAST,
        )
        topic = right_topic if self.which == "right" else left_topic
        self._pub = self._node.create_publisher(HandState, topic, qos)

        # 读取映射（优先实参，其次 JSON）
        self.sim_to_dex3 = sim_to_dex3
        if (self.sim_to_dex3 is None) and mapping_json_path:
            try:
                with open(mapping_json_path, "r") as f:
                    cfg = json.load(f)
                key = "right" if self.which == "right" else "left"
                self.sim_to_dex3 = cfg.get(key) or cfg.get("sim_to_dex3")

                print('!!!!!!!!!')
                print('-----self.sim_to_dex3------')
                print(self.sim_to_dex3)
                print('---------------------------')

                
            except Exception as e:
                print(f"[Dex3State] load mapping json failed: {e}")
                self.sim_to_dex3 = None

        # 没配就取最后 7 个（仅占位，建议你传入明确映射）
        if self.sim_to_dex3 is None:
            self.sim_to_dex3 = list(range(self.num_motors))

        # 简单校验
        bad = [i for i in self.sim_to_dex3 if not isinstance(i, int) or i < 0]
        if bad:
            print(f"[Dex3State][WARN] sim_to_dex3 contains invalid indices: {bad}")

        print(f"[Dex3State] which={self.which}, topic={topic}, rate={self.rate_hz}Hz, sim_to_dex3={self.sim_to_dex3}")

    # ----------------- 小工具 -----------------
    @staticmethod
    def _safe_float(x) -> float:
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return 0.0

    @staticmethod
    def _quat_xyzw_to_wxyz(q):
        try:
            x, y, z, w = [float(q[i]) for i in range(4)]
            return [w, x, y, z]
        except Exception:
            return [1.0, 0.0, 0.0, 0.0]

    # ----------------- 对外接口 -----------------
    def publish(self, env) -> None:
        # 限频
        if self.rate_hz > 0.0:
            now = time.time()
            if (now - self._last_pub_t) < (1.0 / self.rate_hz):
                return
            self._last_pub_t = now

        # 读取仿真状态
        try:
            robot = env.scene["robot"]
            q  = robot.data.joint_pos[0]    # [N]
            dq = robot.data.joint_vel[0]    # [N]
            # 可选：读取 root（用于 IMU 占位）
            root = robot.data.root_state_w[0]    # [13] -> xyz(3) + quat(xyzw)(4) + linvel(3) + angvel(3)
            root_quat_xyzw = root[3:7]
            root_angvel = root[10:13]
        except Exception as e:
            print(f"[Dex3State] read sim failed: {e}")
            return

        # 采样 7 路手指关节
        vals_q, vals_dq = [], []

        for i_sim in self.sim_to_dex3[: self.num_motors]:
            if 0 <= i_sim < len(q):
                vals_q.append(self._safe_float(q[i_sim]))
                vals_dq.append(self._safe_float(dq[i_sim]))
            else:
                vals_q.append(0.0)
                vals_dq.append(0.0)
        # 组 HandState
        msg = HandState()

        # 电机状态（7 路）
        msg.motor_state = [MotorState() for _ in range(self.num_motors)]
        for i in range(self.num_motors):
            ms = msg.motor_state[i]
            ms.mode = 1
            ms.q = vals_q[i]
            ms.dq = vals_dq[i]
            ms.ddq = 0.0
            ms.tau_est = 0.0
            ms.temperature = [25, 25]
            ms.vol = 0.0
            ms.sensor = [0, 0]
            ms.motorstate = 0

        # 压力传感器（仿真没有 → 填空/默认）
        try:
            msg.press_sensor_state = [PressSensorState() for _ in range(self.num_motors)]
        except Exception:
            # 某些版本的 msg 可能没有该类型，忽略即可
            msg.press_sensor_state = []

        # IMU（用 root 近似，或默认单位四元数 + 零角速度）
        qwxyz = self._quat_xyzw_to_wxyz(root_quat_xyzw) if root_quat_xyzw is not None else [1,0,0,0]
        msg.imu_state = IMUState()
        msg.imu_state.quaternion   = [float(v) for v in qwxyz]
        msg.imu_state.gyroscope    = [self._safe_float(v) for v in root_angvel] if root_angvel is not None else [0.0, 0.0, 0.0]
        msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
        msg.imu_state.rpy = [0.0, 0.0, 0.0]     # 如需，可以根据四元数计算 RPY；仿真占位可留 0
        msg.imu_state.temperature = 25

        # 电源/错误（默认）
        msg.power_v = 12.0
        msg.power_a = 0.0
        msg.system_v = 12.0
        msg.device_v = 12.0
        msg.error = [0, 0]
        msg.reserve = [0, 0]

        self._pub.publish(msg)