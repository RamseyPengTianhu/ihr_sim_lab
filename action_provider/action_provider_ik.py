# dds_ik_action_provider.py
import json, threading
from typing import Optional, Dict, Any
import numpy as np
import torch
from action_provider.action_base import ActionProvider

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# PinkIK 正确用法：cfg 里给 FrameTask，运行时改 task.target，然后 compute()
from isaaclab.controllers import PinkIKController, PinkIKControllerCfg
# from isaaclab.controllers.pink_ik_cfg import FrameTask
from pink.tasks import FrameTask
# import pinocchio as pin           # ✅ 构造 SE3 需要
# from pink.utils import SE3   # ✅ 用 Pink 内置的 SE3



class DDSIKActionProvider(ActionProvider):
    def __init__(self, env, args_cli):
        super().__init__("DDSIKActionProvider")
        self.env = env
        self.device = env.device

        # --- 机器人/关节信息 ---
        self.robot = self.env.scene["robot"]
        self.joint_names = list(self.robot.data.joint_names)
        self.num_joints = len(self.joint_names)

        q0 = torch.as_tensor(self.robot.data.joint_pos, device=self.device)
        q0 = q0[0] if q0.ndim == 2 else q0
        self.last_q = q0.clone().to(torch.float32)

        limits = torch.as_tensor(self.robot.data.joint_pos_limits, device=self.device, dtype=torch.float32)
        self.joint_lower = limits[0, :, 0]
        self.joint_upper = limits[0, :, 1]

        # --- CLI / DDS 参数 ---
        self.dds_domain = getattr(args_cli, "dds_domain", 1)
        self.dds_nic    = getattr(args_cli, "dds_nic", "")
        self.ik_topic   = getattr(args_cli, "ik_topic", "/ee_target")
        self.ee_key     = getattr(args_cli, "ik_ee_frame", "right_hand")    # JSON key
        self.ee_link    = getattr(args_cli, "ik_ee_link", "right_hand_tcp") # URDF 链接名

        # --- 速率/滤波 ---
        self.max_qdot = torch.full((self.num_joints,), 2.0, device=self.device, dtype=torch.float32)
        for i, nm in enumerate(self.joint_names):
            if ("shoulder" in nm) or ("elbow" in nm) or ("wrist" in nm):
                self.max_qdot[i] = 4.0
        self.alpha_lpf = 0.25

        # --- DDS 订阅 ---
        self._ee_cache = None
        self._ee_lock  = threading.Lock()
        self._init_ee_target_subscriber(topic=self.ik_topic, domain=self.dds_domain, nic=self.dds_nic)
        print(f"[{self.name}] subscribe {self.ik_topic} ok, domain={self.dds_domain}, nic='{self.dds_nic}', ee_key='{self.ee_key}', ee_link='{self.ee_link}'")

        # ========== PinkIK：用 FrameTask + Cfg 初始化（没有 add_task / set_task_target / solve） ==========
        # 1) 声明一个任务并保存引用：运行时要改它的 target
        self.ee_link = getattr(args_cli, "ik_ee_link", "right_hand_palm_link")

        # 保存为实例属性，后面要改 target
        self.ee_task = FrameTask(
            self.ee_link,          # 第一个位置参数：frame 名
            position_cost=1.0,
            orientation_cost=1.0,
        )

        cfg = PinkIKControllerCfg()
        cfg.urdf_path = "/home/tianhup/Desktop/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
        cfg.mesh_path = "/home/tianhup/Desktop/xr_teleoperate/assets/g1"
        cfg.joint_names = list(self.robot.data.joint_names)
        cfg.variable_input_tasks = [self.ee_task]   # 把实例任务放进去
        cfg.fixed_input_tasks = []
        cfg.show_ik_warnings = True

        self.ik = PinkIKController(cfg, str(self.device))

    # ---------------- DDS 订阅 ----------------
    @staticmethod
    def _find_string_idl():
        for modname in [
            "unitree_sdk2py.idl.default",
            "unitree_sdk2py.idl.unitree_common.msg.dds_",
            "unitree_sdk2py.idl.unitree_api.msg.dds_",
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        ]:
            try:
                mod = __import__(modname, fromlist=['*'])
                for name in dir(mod):
                    if name.lower().endswith("string_"):
                        T = getattr(mod, name)
                        if "data" in getattr(T, "__annotations__", {}):
                            return T
            except Exception:
                pass
        raise RuntimeError("String_ IDL not found in unitree_sdk2py.")

    def _init_ee_target_subscriber(self, topic: str, domain: int, nic: str):
        try:
            ChannelFactoryInitialize(domain, nic) if nic else ChannelFactoryInitialize(domain)
            String_ = self._find_string_idl()
            self._ee_sub = ChannelSubscriber(topic, String_)
            self._ee_sub.Init(self._on_ee_target, 32)
        except Exception as e:
            print(f"[{self.name}] subscribe {topic} failed: {e}")
            self._ee_sub = None

    def _on_ee_target(self, msg):
        try:
            data = json.loads(msg.data)
            with self._ee_lock:
                self._ee_cache = data
        except Exception as e:
            print(f"[{self.name}] parse /ee_target failed: {e} raw={getattr(msg,'data',None)}")

    # ---------------- Utils ----------------
    @staticmethod
    def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
        return q / (q.norm(p=2) + 1e-8)

    def _slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        q0 = self._normalize_quat(q0)
        q1 = self._normalize_quat(q1)
        dot = float((q0 * q1).sum().item())
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return self._normalize_quat((1 - t) * q0 + t * q1)
        theta = torch.acos(torch.clamp(torch.tensor(dot, device=self.device), -1.0, 1.0))
        s0 = torch.sin((1 - t) * theta) / torch.sin(theta)
        s1 = torch.sin(t * theta) / torch.sin(theta)
        return self._normalize_quat(s0 * q0 + s1 * q1)

    def _first_order_lpf(self, target: torch.Tensor, prev: torch.Tensor, alpha: float):
        return alpha * target + (1 - alpha) * prev

    def _apply_rate_limit(self, q_des: torch.Tensor, q_prev: torch.Tensor, dt: float):
        dq = q_des - q_prev
        dq = torch.clamp(dq, -self.max_qdot * dt, self.max_qdot * dt)
        return q_prev + dq

    def _world_to_root(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        # 先用平移简化；若抓取时姿态不准再补旋转
        root_state = self.robot.data.root_state_w  # [N,13]
        root_pos = torch.tensor(root_state[0, :3], device=self.device, dtype=torch.float32)
        pos_local = pos_w - root_pos
        return pos_local, quat_w

    # ---------------- 读取目标 ----------------
    def _read_cartesian_goals_from_dds(self) -> Dict[str, Dict[str, Any]]:
        goals = {}
        try:
            with self._ee_lock:
                cart = None if self._ee_cache is None else dict(self._ee_cache)
            if not cart:
                return goals

            frame = cart.get("frame", "world")

            def grab(key):
                node = cart.get(key, None)
                if node and ("pos" in node) and ("quat" in node):
                    p = torch.tensor(node["pos"], device=self.device, dtype=torch.float32)
                    q = torch.tensor(node["quat"], device=self.device, dtype=torch.float32)  # 默认 wxyz
                    p_prev = getattr(self, f"_{key}_pos_prev", p)
                    q_prev = getattr(self, f"_{key}_quat_prev", q)
                    p_f = self._first_order_lpf(p, p_prev, self.alpha_lpf)
                    q_f = self._slerp(q_prev, q, self.alpha_lpf)
                    setattr(self, f"_{key}_pos_prev", p_f)
                    setattr(self, f"_{key}_quat_prev", q_f)
                    if frame == "world":
                        p_f, q_f = self._world_to_root(p_f, q_f)
                    return {"pos": p_f, "quat": self._normalize_quat(q_f)}
                return None

            for k in ["left_hand", "right_hand", "pelvis", "left_foot", "right_foot"]:
                g = grab(k)
                if g is not None:
                    goals[k] = g
        except Exception as e:
            print(f"[{self.name}] read DDS failed: {e}")
        return goals

    # ---------------- 主接口 ----------------
    def get_action(self, env) -> Optional[torch.Tensor]:
        dt = env.physics_dt
        q_prev = self.last_q

        goals = self._read_cartesian_goals_from_dds()
        if goals and (self.ee_key in goals):
            g = goals[self.ee_key]
            try:
                # 目标位姿 → numpy（Pink 期望 numpy）
                # 目标位姿 → numpy
                pos_np  = g["pos"].detach().cpu().numpy().astype(np.float64)   # (3,)
                quat_np = g["quat"].detach().cpu().numpy().astype(np.float64)  # (w, x, y, z)
                print('pos_np:',pos_np)
                print('quat_np:',quat_np)
                # 小工具：wxyz → 3x3 旋转矩阵（纯 numpy）
                def quat_wxyz_to_rot(q):
                    w, x, y, z = q
                    ww, xx, yy, zz = w*w, x*x, y*y, z*z
                    R = np.array([
                        [ww + xx - yy - zz, 2*(x*y - w*z),   2*(x*z + w*y)],
                        [2*(x*y + w*z),     ww - xx + yy - zz, 2*(y*z - w*x)],
                        [2*(x*z - w*y),     2*(y*z + w*x),   ww - xx - yy + zz]
                    ], dtype=np.float64)
                    return R

                R_np = quat_wxyz_to_rot(quat_np)
                print('R_np:',R_np)
                print('self.ee_link:',self.ee_link)
                # 关键：从 Pink 当前配置取出该 frame 的世界位姿，复制后改写
                T_curr = self.ik.pink_configuration.get_transform_frame_to_world(self.ee_link)  # Pink 自带
                T_tgt  = T_curr.copy()                         # 需要是 Pink 的 Transform，具备 .copy()
                T_tgt.translation = pos_np                     # 平移: (3,)
                T_tgt.rotation    = R_np                       # 旋转: (3,3) 矩阵
                print('T_tgt:',T_tgt)

                # 写回任务目标（注意：这里用的是我们在 __init__ 里保存的 self.ee_task）
                self.ee_task.set_target(T_tgt)

                # 求解
                q_des_np = self.ik.compute(q_prev.detach().cpu().numpy().astype(np.float64), dt)
                q_des = torch.as_tensor(q_des_np, device=self.device, dtype=torch.float32)

                # 限幅 + 限速
                q_des = torch.clamp(q_des, self.joint_lower, self.joint_upper)
                q_des = self._apply_rate_limit(q_des, q_prev, dt)
                self.last_q = q_des

                return q_des.unsqueeze(0)

            except Exception as e:
                print(f"[{self.name}] IK compute failed: {e}")

        # 无目标/失败 → 保持上一帧
        return q_prev.unsqueeze(0)

    def cleanup(self):
        pass


# for simulation: python sim_main.py   --task Isaac-Waste-Sort-G129-Dex3   --device cuda:0   --action_source dds_ik   --ik_topic /ee_target   --ik_ee_frame right_hand   --dds_domain 1   --dds_nic eno2   --enable_dex3_dds   --robot_type g129 --enable_cameras
# for dds (new terminal)L python unitree_sdk2_python/example/g1/high_level/pub_ee_target.py --domain 1 --nic eno2 --topic /ee_target --frame world --side right --hz 50