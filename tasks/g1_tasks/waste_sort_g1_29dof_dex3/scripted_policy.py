# tasks/g1_tasks/waste_sort_g1_29dof_dex3/scripted_policy.py
import numpy as np

class ScriptedPolicy:
    def __init__(self, env):
        # 简单小摆动：只动前3个关节做正弦摆
        self.t = 0

    def __call__(self, env):
        # 当前关节位置

        print("[policy] step", time.time())

        robot = env.scene["robot"]
        q = robot.data.joint_pos.detach().cpu().numpy()   # (num_envs, 43)
        num_envs, dof = q.shape

        # 目标 = 当前位置拷贝
        q_des = q.copy()

        # 给前三个关节加个正弦
        amp = np.array([0.2, 0.2, 0.2])  # 幅度
        freq = 0.5                       # 频率 Hz（仿真步长0.008 → 每秒约125步）
        phase = self.t * 2 * np.pi * freq / 125.0
        delta = amp * np.sin(phase)

        q_des[:, 0:3] = q[:, 0:3] + delta

        # 计时
        self.t += 1

        # 打印一次看是否执行
        if self.t % 200 == 0:
            print("[policy] step", self.t)

        return q_des