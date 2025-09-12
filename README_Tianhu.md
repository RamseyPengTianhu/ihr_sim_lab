# Action Provider 使用说明

本仓库提供多种 **ActionProvider** 实现，用于 **Isaac Lab** 与 **DDS / 本地策略 / IK** 等多种控制方式的对接。

---

## 模块说明

### 1. DDSActionProvider
- **控制范围**：上肢 14 个关节（双臂），可选手爪 / 灵巧手 / INSPIRE。
- **数据来源**：通过 `dds_manager` 从 DDS 订阅（集中式路由模式）。
- **场景适用**：只需上肢动作时使用。

---

### 2. DDSWholebodyActionProvider
- **控制范围**：全身 29 个关节（腿 + 腰 + 臂）。
- **数据来源**：内部直接启动 DDS 订阅者（各子模块直连模式）。
- **场景适用**：需要全身控制的任务。

---

### 3. TrajectoryActionProvider
- **功能**：本地生成简单轨迹（如挥手演示）。
- **用途**：调试 / 演示非常方便，无需外部数据源。

---

### 4. PolicyActionProvider
- **功能**：占位（stub），当前无加载策略权重逻辑。
- **你需要补充**：
  - 读取 obs
  - 前向推理
  - 输出 action
- **场景适用**：直接用训练好的 RL 策略控制机器人。

---

### 5. DDSIKActionProvider（新增）
- **功能**：结合 DDS 接收的笛卡尔空间目标 + Pink IK 模块求解关节位置。
- **特点**：在 Isaac Lab 端直接做 IK 计算，输入为末端位置 + 四元数，输出关节角。
- **相关代码**：
  - IK 逻辑：`action_provider/action_provider_ik.py`
  - DDS 输入示例：`unitree_sdk2_python/example/g1/high_level/pub_ee_target.py`

#### DDSIK 输入 JSON 格式示例
DDSIKActionProvider 订阅的 `/ee_target` 消息应包含：
```json
全局坐标
{
  "frame": "world",
  "pos": [0.35, -0.20, 0.90],
  "quat": [0, 0, 0, 1],
  "side": "right"
}
Local 坐标
{
  "frame": "root",
  "pos": [0.35, -0.20, 0.90],
  "quat": [0, 0, 0, 1],
  "side": "right"
}
```


---

frame：参考系（常用 "world"）

pos：目标末端位置 (x, y, z)，单位 m

quat：目标末端朝向 (x, y, z, w)

side："right" 或 "left"





#### DDSIK 输入 JSON 格式示例
启动方式
DDSActionProvider（上肢）

Simulation 端：

python sim_main.py \
  --task Isaac-Waste-Sort-G129-Dex3 \
  --device cuda:0 \
  --action_source dds \
  --dds_domain 1 \
  --dds_nic eno2 \
  --enable_dex3_dds \
  --robot_type g129 \
  --enable_cameras


DDS 端：

python unitree_sdk_python/example/g1/high_level/pub_lowcmd.py

DDSWholebodyActionProvider（全身）

启动方式类似于 DDSActionProvider，只需将 --action_source 改为 dds_wholebody。

TrajectoryActionProvider（轨迹演示）

本地生成轨迹，不需要 DDS 端启动。
启动方式：

python sim_main.py \
  --task Isaac-Waste-Sort-G129-Dex3 \
  --device cuda:0 \
  --action_source trajectory

PolicyActionProvider（策略）

占位模式，目前需要你自己实现加载策略和推理。
启动方式：

python sim_main.py \
  --task Isaac-Waste-Sort-G129-Dex3 \
  --device cuda:0 \
  --action_source policy

DDSIKActionProvider（新）

Simulation 端：

python sim_main.py \
  --task Isaac-Waste-Sort-G129-Dex3 \
  --device cuda:0 \
  --action_source dds_ik \
  --ik_topic /ee_target \
  --ik_ee_frame right_hand \
  --dds_domain 1 \
  --dds_nic eno2 \
  --enable_dex3_dds \
  --robot_type g129 \
  --enable_cameras


DDS 端（新终端）：

python unitree_sdk2_python/example/g1/high_level/pub_ee_target.py \
  --domain 1 \
  --topic /ee_target \
  --frame world \
  --side right \
  --hz 50


修改 IK 逻辑请参见 action_provider/action_provider_ik.py
修改输入端逻辑请参见 unitree_sdk2_python/example/g1/high_level/pub_ee_target.py


DDSROSIKActionProvider（新）

Simulation 端：

source /opt/ros/humble/setup.bash
source ~/Desktop/unitree_ros2/cyclonedds_ws/install/setup.bash
conda activate isaaclab310
python sim_main.py --device cuda:0 --enable_cameras   --task Isaac-Waste-Sort-G129-Dex3 --robot_type g129 --action_source ros_ik
DDS 端（新终端）：
source /opt/ros/humble/setup.bash
source ~/Desktop/unitree_ros2/cyclonedds_ws/install/setup.bash
conda activate isaaclab310
ros2 topic pub -r 10 /ee_target geometry_msgs/PoseStamped "{
  header: {frame_id: 'base'},
  pose: {position: {x: 0.35, y: 0.0, z: 0.00},
         orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
}"
DDS 端 机械手
ros2 topic pub -r 30 /dex3/right/cmd unitree_hg/msg/HandCmd "{
  motor_cmd: [
    {mode: 1, q: 0.5, dq: 0.0, tau: 0.8, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: 0.6, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: 0.2, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: 0.5, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: 0.5, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: -0.5, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0},
    {mode: 1, q: -0.5, dq: 0.0, tau: 0.0, kp: 1.0, kd: 0.05, reserve: 0}
  ]
}"
第三个是大拇指yaw

官方任务场景

宇树官方提供 4 个基础任务（每个任务针对 3 类机械手搭建）：

pick_place_cylinder_g1_29dof

pick_place_redblock_g1_29dof

pick_refblock_into_drawer_g1_29dof

stack_rgyblock_g1_29dof

ATEC 比赛新增任务

Task1：waste_sort_g1_29dof_dex3

功能：分类垃圾

当前仅支持宇树三指灵巧手

文件路径：tasks/g1_tasks/waste_sort_g1_29dof_dex3

残留进程检查与清理

结束 Isaac Lab 后建议检查是否有残留进程：

ps -ef | egrep "isaac|IsaacLab|omni|kit|sim_main.py|python.*isaac" | grep -v grep


如有残留，执行：

sudo kill -9 <PID>


其中 <PID> 为上一步查到的进程号。




DDSROSActionProvider（新）

Simulation 端：

source /opt/ros/humble/setup.bash
source ~/Desktop/unitree_ros2/cyclonedds_ws/install/setup.bash
conda activate isaaclab310
python sim_main.py --device cuda:0 --enable_cameras   --task Isaac-Waste-Sort-G129-Dex3 --robot_type g129 --action_source ros

ros2端：
# 先清掉旧配置
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source ~/Desktop/unitree_ros2/cyclonedds_ws/install/setup.bash
# 如果要用 unitree_ros2 里的消息/例程，再：
source ~/Desktop/unitree_ros2/unitree_ros2_ws/install/setup.bash

source /opt/ros/humble/setup.bash
source ~/Desktop/unitree_ros2/cyclonedds_ws/install/setup.bash
conda activate isaaclab310
source ~/Desktop/unitree_ros2/setup_default.sh # 不指定网卡
./install/unitree_ros2_example/bin/g1_arm7_control