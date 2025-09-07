# ros_publisher/dex3_hand_cmd_pub.py
import json
from typing import List, Optional, Sequence, Union

import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from unitree_hg.msg import HandCmd, MotorCmd


TensorLike = Union[torch.Tensor, Sequence[float]]


class Dex3HandPublisher:
    """
    Dex3 手部命令发布/融合工具：
      - 订阅 /dex3/{right,left}/cmd 的 HandCmd（可选），缓存最近 7 路手指 q（弧度）
      - 将手指 q 注入 full_action（按 sim_to_dex3 映射），返回更新后的 (1, N)
      - 发布 HandCmd（从 full_action 抽取或直接给 7 维）

    话题:
      - 右手发布: /dex3/right/cmd
      - 左手发布: /dex3/left/cmd
      - 订阅同一话题以接收外部手命令（可配置关闭）
    """

    def __init__(
        self,
        node,                              # rclpy Node
        which: str = "right",              # "right" or "left"
        right_topic: str = "/dex3/right/cmd",
        left_topic: str = "/dex3/left/cmd",
        sim_to_dex3: Optional[List[int]] = None,   # 长度=7：full_action 索引 -> 手7路顺序
        mapping_json_path: Optional[str] = None,   # JSON: {"right":[...], "left":[...]} 或 {"sim_to_dex3":[...]}
        # 接收外部ROS手命令（订阅）
        subscribe_external_cmd: bool = True,
        # 手目标单位与安全约束
        hand_in_degree: bool = False,              # 外部/内部手目标是否用“度”（会转弧度）
        joint_lower_7: Optional[Sequence[float]] = None,  # 7 维下限（弧度）
        joint_upper_7: Optional[Sequence[float]] = None,  # 7 维上限（弧度）
        rate_limit_rad_per_s: float = 4.0,         # 限速：最大角速度 (rad/s)
        dt: float = 0.005,                         # 控制周期
        debug: bool = False,
    ):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        self._node = node
        self.which = which.lower()
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # 话题 & 发布器
        self._pub_topic = right_topic if self.which == "right" else left_topic
        self._pub = self._node.create_publisher(HandCmd, self._pub_topic, qos)

        # 可选：订阅外部手命令（同一话题）
        self._sub = None
        if subscribe_external_cmd:
            self._sub = self._node.create_subscription(
                HandCmd, self._pub_topic, self._cb_external_hand_cmd, qos
            )

        # ---- 映射加载 ----
        self.sim_to_dex3 = sim_to_dex3
        if (self.sim_to_dex3 is None) and mapping_json_path:
            try:
                with open(mapping_json_path, "r") as f:
                    cfg = json.load(f)
                key = "right" if self.which == "right" else "left"
                self.sim_to_dex3 = cfg.get(key) or cfg.get("sim_to_dex3")
            except Exception as e:
                print(f"[Dex3] load mapping json failed: {e}")
        if self.sim_to_dex3 is None:
            self.sim_to_dex3 = list(range(7))
        if len(self.sim_to_dex3) != 7:
            print(f"[Dex3][WARN] sim_to_dex3 len={len(self.sim_to_dex3)}, padding/trunc to 7.")
            pad_val = self.sim_to_dex3[-1] if self.sim_to_dex3 else 0
            self.sim_to_dex3 = (self.sim_to_dex3 + [pad_val])[:7]

        # ---- 参数与缓存 ----
        self.hand_in_degree = hand_in_degree
        self.joint_lower_7 = (
            torch.as_tensor(joint_lower_7, dtype=torch.float32).view(-1)
            if joint_lower_7 is not None else None
        )
        self.joint_upper_7 = (
            torch.as_tensor(joint_upper_7, dtype=torch.float32).view(-1)
            if joint_upper_7 is not None else None
        )
        self.rate_limit = float(rate_limit_rad_per_s)
        self.dt = float(dt)
        self.debug = bool(debug)

        # 手目标缓存（弧度）
        self.q7_cache: Optional[torch.Tensor] = None   # 最近收到/设置的 7 路目标
        self.q7_valid: bool = False                    # 是否有“未消费”的新目标
        self._last_q7: Optional[torch.Tensor] = None   # 上次已注入/已发布的 7 路，用于限速

        if self.debug:
            print(f"[Dex3] pub={self._pub_topic}, sub={bool(self._sub)}, which={self.which}, sim_to_dex3={self.sim_to_dex3}")

    # ========== 外部 ROS 订阅回调：缓存 q7 ==========
    def _cb_external_hand_cmd(self, msg: HandCmd):
        qs = [mc.q for mc in (msg.motor_cmd or [])]
        # 容错：长度凑到7
        if len(qs) < 7:
            tail = qs[-1] if qs else 0.0
            qs = qs + [tail] * (7 - len(qs))
        elif len(qs) > 7:
            qs = qs[:7]

        q7 = torch.as_tensor(qs, dtype=torch.float32).view(-1)  # 认为外部单位=弧度（若你外部发的是度，请把 hand_in_degree=True 并用 set_q7 流程）
        if self.hand_in_degree:
            q7 = torch.deg2rad(q7)

        # 限位
        if (self.joint_lower_7 is not None) and (self.joint_upper_7 is not None):
            lo = self.joint_lower_7.to(q7.device, q7.dtype)
            hi = self.joint_upper_7.to(q7.device, q7.dtype)
            q7 = torch.clamp(q7, lo, hi)

        self.q7_cache = q7
        self.q7_valid = True
        if self.debug:
            self._node.get_logger().info(f"[Dex3] recv external q7(rad)={q7.tolist()}")

    # ========== 公共接口 ==========
    def set_q7(self, q7: TensorLike) -> None:
        """内部设置 7 维手目标；若 hand_in_degree=True 会转换为弧度。"""
        q7t = _to_tensor_1d(q7, dtype=torch.float32)
        if q7t.numel() != 7:
            raise ValueError(f"q7 must have 7 elems, got {q7t.numel()}")
        if self.hand_in_degree:
            q7t = torch.deg2rad(q7t)
        # 限位
        if (self.joint_lower_7 is not None) and (self.joint_upper_7 is not None):
            lo = self.joint_lower_7.to(q7t.device, q7t.dtype)
            hi = self.joint_upper_7.to(q7t.device, q7t.dtype)
            q7t = torch.clamp(q7t, lo, hi)
        self.q7_cache = q7t
        self.q7_valid = True
        if self.debug:
            print(f"[Dex3][set_q7] q7(rad)={q7t.tolist()}")

    def publish_q7(self, q7: Optional[TensorLike] = None) -> None:
        """
        直接发布 7 维手目标（跳过 full_action）。
        若 q7=None 则使用缓存/上一帧。
        """
        if q7 is not None:
            self.set_q7(q7)
        if not self.q7_valid and self._last_q7 is None:
            if self.debug:
                print("[Dex3][publish_q7] no q7 to publish.")
            return
        q7t = self._prepare_q7_for_send()
        _publish_hand_cmd(self._pub, q7t, debug=self.debug)
        self._last_q7 = q7t.detach().cpu()
        self.q7_valid = False

    def inject_into_full_action(self, full_action: TensorLike) -> torch.Tensor:
        """
        仅做“注入”：
          - 若缓存中有新 q7，则将其（限位/限速后）写回 full_action 对应的 7 个索引
          - 返回更新后的 (1, N) 张量
          - 不发布 HandCmd
        """
        fa = _to_tensor_2d(full_action)
        if not self.q7_valid or (self.q7_cache is None):
            return fa  # 无新手目标，不改动

        q7t = self._prepare_q7_for_send(device=fa.device, dtype=fa.dtype)
        fa = fa.clone()
        fa[0, self.sim_to_dex3] = q7t

        self._last_q7 = q7t.detach().cpu()
        self.q7_valid = False

        if self.debug:
            view = fa[0, self.sim_to_dex3].detach().cpu().numpy()
            print(f"[Dex3][inject_only] hand slice: min={view.min():.3f} max={view.max():.3f}")
        return fa

    def publish_from_full_action(
        self,
        full_q_tensor: TensorLike,
        q7_override: Optional[TensorLike] = None,  # 可传入本次覆盖的 7 维（度/弧度按 hand_in_degree 决定）
        inject_cached_q7: bool = True,
        return_updated: bool = True,
    ) -> torch.Tensor:
        """
        组合流程：
          1) 可选：q7_override → set_q7()
          2) 可选：将缓存的 q7 注入 full_action（限位/限速）
          3) 从（可能已覆盖的）full_action 中抽取 7 维并发布 HandCmd
          4) 返回（更新后的）full_action
        """
        fa = _to_tensor_2d(full_q_tensor)

        if q7_override is not None:
            self.set_q7(q7_override)

        # 1) 注入缓存的 q7
        if inject_cached_q7 and self.q7_valid and (self.q7_cache is not None):
            q7t = self._prepare_q7_for_send(device=fa.device, dtype=fa.dtype)
            fa = fa.clone()
            fa[0, self.sim_to_dex3] = q7t
            self._last_q7 = q7t.detach().cpu()
            self.q7_valid = False
            if self.debug:
                view = fa[0, self.sim_to_dex3].detach().cpu().numpy()
                print(f"[Dex3][inject] hand slice after inject: min={view.min():.3f} max={view.max():.3f}")

        # 2) 抽取 7 维并发布
        hand_vals = fa[0, self.sim_to_dex3].detach().cpu()
        _publish_hand_cmd(self._pub, hand_vals, debug=self.debug)
        self._last_q7 = hand_vals.clone()

        # 3) 返回
        return fa if return_updated else _to_tensor_2d(full_q_tensor)

    # ========== 内部工具 ==========
    def _prepare_q7_for_send(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        将缓存的 q7 做单位转换、限位、限速，返回 7 维弧度向量（在指定 device/dtype）。
        """
        if self.q7_cache is None and self._last_q7 is None:
            q7 = torch.zeros(7, dtype=torch.float32)
        elif self.q7_cache is None:
            q7 = self._last_q7.clone().float()
        else:
            q7 = self.q7_cache.clone().float()

        # 限位
        if (self.joint_lower_7 is not None) and (self.joint_upper_7 is not None):
            lo = self.joint_lower_7.to(q7.device, q7.dtype)
            hi = self.joint_upper_7.to(q7.device, q7.dtype)
            q7 = torch.clamp(q7, lo, hi)

        # 限速
        if self._last_q7 is not None:
            prev = self._last_q7.to(q7.device, q7.dtype)
            max_delta = float(self.rate_limit) * float(self.dt)
            delta = torch.clamp(q7 - prev, -max_delta, +max_delta)
            q7 = prev + delta

        # 目标设备/精度
        if device is not None or dtype is not None:
            q7 = q7.to(device=device or q7.device, dtype=dtype or q7.dtype)
        return q7


# ---------------- 辅助函数 ----------------
def _to_tensor_1d(x: TensorLike, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().clone()
        if t.ndim == 0:
            t = t.view(1)
        elif t.ndim > 1:
            t = t.view(-1)
        return t.to(dtype=dtype)
    else:
        return torch.as_tensor(x, dtype=dtype).view(-1)


def _to_tensor_2d(x: TensorLike, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().clone().to(dtype=dtype)
        if t.ndim == 1:
            t = t.view(1, -1)
        elif t.ndim == 2:
            return t
        else:
            t = t.view(1, -1)
        return t
    else:
        t = torch.as_tensor(x, dtype=dtype)
        return t.view(1, -1)


def _publish_hand_cmd(pub, q7_tensor: torch.Tensor, debug: bool = False) -> None:
    """
    将 7 维弧度向量发布为 HandCmd。
    """
    q7 = q7_tensor.detach().cpu().view(-1).tolist()
    if len(q7) != 7:
        raise ValueError(f"hand cmd must have 7 elems, got {len(q7)}")

    msg = HandCmd()
    msg.motor_cmd = [MotorCmd() for _ in range(7)]
    for i in range(7):
        m = msg.motor_cmd[i]
        # 注意：以下模式/增益需按你的真机配置调整
        m.mode = 0x11
        m.q    = float(q7[i])   # 目标位置（弧度）
        m.dq   = 0.0
        m.tau  = 0.0
        m.kp   = 1.0
        m.kd   = 0.05

    try:
        pub.publish(msg)
        if debug:
            print(f"[Dex3][publish] q7(rad)={q7}")
    except Exception as e:
        print(f"[Dex3] publish failed: {e}")
