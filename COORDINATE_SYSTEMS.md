# 坐标系与坐标变换完整手册

> 涵盖：(1) PushT Demo 的所有坐标系 (2) Super 外科数据集的所有坐标系 (3) 两者之间的坐标系转换过程。
>
> 这份文档按「场景里有什么 → 各自的坐标系是什么 → 它们之间怎么转换」的顺序组织。

---

## 第一部分：基础知识

### 1.1 什么是坐标系

坐标系 = 一个原点 + 三个互相垂直的方向轴 (X, Y, Z)。

同一个 3D 点在不同坐标系里数字不同，但它本身没动。

本文所有坐标系都遵循 OpenCV/相机标准：
```
X→右   Y→下   Z→前（远离相机深入场景）
```

### 1.2 什么是坐标变换

把 A 坐标系里的坐标 (x_A, y_A, z_A) 变成 B 坐标系里的坐标 (x_B, y_B, z_B)。

用 4×4 齐次矩阵 `X_AB` 表示：`p_B = X_AB × p_A`

**读法**：`X_AB` = "从 A 变到 B"。A 是「从哪来」，B 是「到哪去」。

```python
# 4×4 矩阵 = 旋转(左上3×3) + 平移(右上3×1)
#
#  [ r11 r12 r13 tx ]   [ x_A ]   [ x_B ]
#  [ r21 r22 r23 ty ] × [ y_A ] = [ y_B ]
#  [ r31 r32 r33 tz ]   [ z_A ]   [ z_B ]
#  [  0   0   0  1  ]   [  1  ]   [  1  ]
```

**本文所有变换符号**：

| 符号 | 含义 | 举例 |
|------|------|------|
| `X_WB` | Body → World | 把物体局部坐标转成世界坐标 |
| `X_WC` | Camera → World | 把相机坐标转成世界坐标 |
| `X_CW` | World → Camera | X_WC 的逆矩阵，渲染时用 |
| `I` | 单位矩阵 | 啥也不变：`I × p = p` |
| `T` | 自定义变换 | 本文特指「桌面坐标变换」|

### 1.3 相机投影（3D → 2D）

把相机坐标系里的 3D 点 (X, Y, Z) 投影到图像像素 (u, v)：

```
u = fx · (X / Z) + cx
v = fy · (Y / Z) + cy
```

参数 `fx, fy, cx, cy` 组成内参矩阵 K：

```
    [ fx   0  cx ]
K = [  0  fy  cy ]
    [  0   0   1 ]
```

**直觉**：X/Z 和 Y/Z 是该点在相机前方的「角度」。离相机越远 (Z 越大)，X/Z 越小 → 图像上越靠近中心。

### 1.4 深度反投影（2D → 3D）

如果知道像素 (u, v) 对应的深度 Z，可以反过来求 3D 坐标：

```
X = (u - cx) · Z / fx
Y = (v - cy) · Z / fy
Z = depth[v, u]    ← 从深度图读取
```

---

## 第二部分：PushT Demo 的坐标系

### 2.1 场景

> Panda 机械臂放在桌面上，推一个 T 形方块。3 台固定相机从不同角度拍摄，夹爪上还有 1 台手眼相机。

### 2.2 世界坐标系 (W)

**原点**：桌面中心。**方向**：X→右, Y→前, Z→上。

世界是所有物体的最终参考系。物理仿真在这个坐标系里进行。

```
        Z (上)
        ↑
        |
    ┌───┼───────────────┐
    │   │    桌面(z=0)    │
    │   ┌─────┐          │
    │   │T-block│         │
    │   └─────┘          │
    │      ┌────┐        │
    │      │Panda│       │
    │      └────┘        │
    └────────────────────┘
          世界原点 ●
```

**重力**：`(0, 0, -9.8)` m/s²，沿 -Z 方向（垂直向下，压住物体）。

**地面碰撞面**：`[0, 0, 1, 0]` → 平面方程 `z = 0`。任何物体不能穿透 z=0 平面。

### 2.3 Panda 机械臂 (URDF)

Panda 有 7 个旋转关节。URDF 文件定义了运动学链：

```
world ─[fixed]─→ panda_link0 ─[joint1,revolute]─→ panda_link1 ─[joint2]─→ ... ─[joint7]─→ panda_hand
```

URDF 中 `fixed` 关节把 Panda 基座固定在 world 坐标系中。基座的 `xyz` 和 `rpy` 定义了机械臂在世界中的位置。

每个关节绕自己的 `axis` 旋转角度 `q`。关节角来自 `robots.json`。

### 2.4 T-block 物体的坐标系

T-block 的 JSON 文件 (`tblock.json`) 定义了：

```json
{
  "name": "tblock",
  "X_WB": [[...], [...], [...], [0,0,0,1]],
  "gaussians": { "means": [...], "quats": [...], ... },
  "particles": { "means": [...], "quats": [...], "radii": [...], ... }
}
```

- `X_WB`：Body → World 的变换矩阵。把 tblock 身体坐标转成世界坐标。
- `gaussians.means`：每个高斯在**身体坐标系**里的位置。
- `particles.means`：每个粒子在**身体坐标系**里的位置。

代码中通过 `add_rigid_body()` 将物体加入场景。`X_WB` 的平移部分决定该物体放在世界中的什么位置。

### 2.5 地面物体的坐标系

地面的 JSON 文件 (`ground.json`) 同样定义了 `X_WB`、gaussians、particles。

地面通过 `add_visual_body()` 加入，**body_id=-1**，只参与渲染，不参与物理仿真。物理碰撞由 `ground_plane` 处理。

### 2.6 相机坐标系

Demo 有 4 台相机，代码在 `pusht_embodied.py` 的 `add_static_cameras()`：

```python
def add_static_cameras(builder: VirtualCamerasBuilder):
    K = np.array([[241, 0, 240], [0, 241, 135], [0, 0, 1]])
    camera_data = [
        ("234222302164", X_WC_1),  # X_WC 写死在代码里
        ...
    ]
    for camera_id, X_WC in camera_data:
        builder.add_camera(camera_id, K, X_WC)
```

每台相机有：
- **内参 K**：3×3 矩阵，定义了像素焦距和光心。
- **外参 X_WC**：4×4 矩阵，定义了相机在世界坐标系中的位姿。

**从世界坐标到相机坐标的变换**：`X_CW = inv(X_WC)`。渲染时用 `X_CW` 把世界坐标的高斯投影到相机视角。

### 2.7 Demo 中坐标变换的完整流程（一帧）

```
STEP 1: 读取 robots.json
        得到当前帧的关节角 q[0..6]

STEP 2: 设置 Panda 关节目标
        env.set_robot_desired_q(0, q)
        → 物理引擎 PD 控制把关节推向目标

STEP 3: 物理仿真 (physics_step)
        - 碰撞检测
        - XPBD 约束求解
        - 关节 PD 控制
        - 积分更新 body_q（每个刚体的世界位姿）

STEP 4: 更新高斯位姿
        body_q[body_id] × gaussian_local_pose → gaussian_world_pose
        即: 刚体动了 → 绑在上面的高斯也跟着动

STEP 5: 渲染
        对每台相机:
          X_CW = inv(X_WC)
          把高斯世界坐标 → 转到相机坐标 → 通过 K 投影 → 渲染成图像

STEP 6: Visual Forces
        渲染图 vs 真实图 (从 MP4 读的) → 像素差异 → 梯度反传
        → 算出力 → 施加到物理引擎 → 修正物体位置
```

### 2.8 PushT 中的变换链总结

```
URDF 关节角 q ──→ FK前向运动学 ──→ 每个 link 的 X_WB ──→ 高斯世界位姿
                                                              │
相机 X_WC ──→ X_CW = inv(X_WC) ──→ 高斯转到相机坐标 ──→ K 投影 ──→ 渲染图
```

---

## 第三部分：Super 外科数据集的坐标系

### 3.1 场景

> dVRK PSM1 手术器械在腹腔镜相机下方操作软组织。两台立体相机（左眼+右眼）拍摄手术区域。

### 3.2 数据来源与处理链

```
grasp5.bag (ROS 录制)
    │
    ├── 关节状态 (5458条) ──→ robots.json
    │
    ├── 左/右眼原始图像 (1644对, 1920×1080)
    │   │
    │   ├── 立体校正 (stereoRectify) ──→ rectified PNG
    │   │
    │   ├── 拼成 MP4 ──→ stereo_left.mp4 + stereo_right.mp4
    │   │
    │   └── 深度估计 (Python-SuPer) ──→ depth.npy
    │
    ├── SAM2 分割 ──→ tissue/ground mask
    │
    └── 反投影 (深度 + mask) ──→ 3D 点云 ──→ tissue.json, ground.json
                                          │
                                          └──→ 拟合地面平面 → ground_plane.json
```

### 3.3 相机坐标系 (C) — 当前世界坐标系

**原点**：左眼 rectified 相机的光心。

**为什么**：深度反投影出来的 3D 点天然在相机坐标系里。为了省事，直接把世界坐标系设定为和相机坐标系重合（`X_WC = I`）。

```
      相机光心 = 世界原点 (0,0,0)
           ●
          /|
         / |  Z轴（看的方向，深入腹腔）
        /  |
       /   |
      ●    | ← 组织上的点 (X,Y,Z)
     ╱╲   |
    ╱══╲  |
   ╱ 桌面 ╲|
  ━━━━━━━━━━━  桌面是斜的（腹腔镜从侧上方看）
```

### 3.4 组织物体 (tissue) 的坐标系

- `X_WB = I`（单位矩阵）
- 所以**世界坐标 = 身体坐标 = 相机坐标**
- 1976 个点，3D 范围：x[-42,39]mm, y[-32,19]mm, z[71,117]mm

**每个点是怎么来的**：
```
某个像素 (u,v) 在 tissue mask 里
    → 读深度图: Z = depth[v,u] = 比如 0.090m
    → X = (u-860.4) × 0.090 / 1742.8
    → Y = (v-682.3) × 0.090 / 1742.8
    → 保存为 gaussians.means 和 particles.means 的一个元素
```

### 3.5 桌面物体 (ground) 的坐标系

- `X_WB = I`
- 世界坐标 = 相机坐标
- 2000 个点，3D 范围：x[-77,66]mm, y[-52,25]mm, z[88,165]mm

**ground 的特殊之处**：它是平的。构建脚本 `build_super_ground.py` 不是直接用深度点，而是：
1. 用深度点拟合平面方程
2. 在平面上生成均匀网格
3. 筛选（排除组织覆盖区域）
4. 降采样

所以 ground 所有点严格落在同一个平面上。

### 3.6 地面碰撞面的坐标系

```json
{"plane": [0.155925, 0.583985, 0.796648, -0.088997]}
```

这是平面方程 `a·x + b·y + c·z + d = 0` 的系数。

- 法向量：`(0.156, 0.584, 0.797)` — 不垂直于世界 Z 轴，差 37.2°
- 偏移：`0.089`（原点到平面的带符号距离）

**为什么是斜的**：腹腔镜不是垂直向下拍摄的。它在患者上方斜着看手术区域。桌面在相机坐标系里自然是斜的。

### 3.7 左右相机的坐标系

#### 左相机 (stereo_left)

```json
{"X_WC": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

`X_WC_left = I`。因为世界原点就是左相机光心，不需要变换。

#### 右相机 (stereo_right)

```json
{"X_WC": [[1,0,0,-0.0053],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

`X_WC_right` = 左相机坐标系往左平移 5.3mm（立体基线）。

**为什么**：rectified 之后左右眼完全水平对齐。右相机在左相机的 X 轴负方向（baseline = 5.3mm = 0.0053m）。

两台相机的**内参 K 完全相同**（rectified 后的特性）。

### 3.8 Super 数据集坐标系问题

| 问题 | 原因 | 后果 |
|------|------|------|
| 地面碰撞面不水平 | 法向量 (0.156,0.584,0.797) ≠ (0,0,1) | 重力不完全垂直于桌面 |
| 重力方向沿世界 Z 轴 (-Z) | 物理引擎默认 | 和桌面法向量差 37.2° |
| 组织会沿斜面滑落 | 重力分力 ≈ 5.9 m/s² 沿斜面 | 物理仿真不真实 |
| 相机在世界原点 | X_WC=I 是临时措施 | 和实际场景不符 |

**解决**：做桌面坐标变换 → 让桌面变成 z=0 水平面。

---

## 第四部分：桌面坐标变换

### 4.1 目标

把整个场景旋转 + 平移，使得：

1. 桌面法向量 = `(0, 0, 1)`（即桌面变成水平面）
2. 桌面落在 z = 0（即桌面在世界坐标系原点平面上）
3. 重力 `(0, 0, -9.8)` 垂直于桌面

### 4.2 变换矩阵 T

一个 4×4 齐次矩阵 `T = [R | t]`，其中 R 是旋转、t 是平移。

对场景中**每一个 3D 点**做：`p_new = T × p_old`

### 4.3 需要变换的所有东西

| 东西 | 变换前的值 | 变换后的值 | 怎么算 |
|------|-----------|-----------|--------|
| tissue 1976 个 3D 点 | 相机坐标 | 右手系桌面坐标 | `p_new = Rx(180) × T × p_old` |
| ground 2000 个 3D 点 | 相机坐标 | 右手系桌面坐标 | `p_new = Rx(180) × T × p_old` |
| ground_plane 平面方程 | `n·x+d=0`(相机坐标) | `n'·x+d'=0`(桌面坐标) | `n' = R×n`, `d' = d - n'·t` |
| X_WC_left (左相机外参) | I | OpenCV pose = **Rx(180) × T**；cameras.json 存 Blender pose | renderer 内部再转 OpenCV |
| X_WC_right (右相机外参) | I + 基线偏移 | OpenCV pose = Rx(180) × T × 旧值；cameras.json 存 Blender pose | 同左相机 |
| X_WB_tissue | I | I | tissue 点坐标已直接写入右手系 world/local，避免二次变换 |
| X_WB_ground | I | I | ground 点坐标已直接写入右手系 world/local，避免二次变换 |
| PSM 基座位姿 | rectified camera 下 handeye/LND 结果 | Rx(180) × T × 基座_旧 | 写入 URDF fixed joint |

### 4.4 变换矩阵的来源

super2 已计算过这个变换。文件：

```
embodied_gaussians_fixed_super2/data/super/
  grasp5_processed_before_z_axis_realign_20260610_053315/
    super_table_frame_transform.json
```

其中的 `X_table_from_source` 是基础对齐矩阵 T。最终 Demo 使用 `Rx(180) @ T`，其中 `Rx(180)=diag(1,-1,-1,1)`，这样相机位于桌面 z>0 一侧且旋转保持右手系。

### 4.5 变换后的验证

```
右手系变换后的 ground_plane 应该是 [0, 0, 1, 0] 或非常接近。
即: n' = (0,0,1), d' = 0。桌面 = z=0 水平面。
```

### 4.6 完整变换流程示意

```
变换前（相机坐标系 = 世界坐标系）:

        相机 ●
             \
              \  组织
               \ ╱╲
                ╲━━━  斜桌面
重力 ↓↓        /  (不垂直于桌面!)
              /
             ● PSM (待定)

────────────────────────────────────

变换后（桌面坐标系 = 世界坐标系）:

        相机 ●  (新位置：桌面上方斜角)
             \
              \  组织
               \ ╱╲  (稳!)
  ━━━━━━━━━━━━━━━━━  z=0 桌面
重力 ↓↓↓ (垂直桌面!)


                PSM ● (新位置)
```

---

## 第五部分：PSM 机械臂的坐标系

### 5.1 PSM URDF 的运动学链

PSM1 有 7 个活动关节。URDF 定义了关节间的空间关系：

```
world ─[fixed, xyz=-0.25,0,0.5, rpy=0,0,3.14]─→ PSM1_psm_base_link
        └─[yaw,      revolute]─→ PSM1_outer_yaw_link
          └─[pitch,    revolute]─→ PSM1_outer_pitch_link
            └─[insertion,PRISMATIC]─→ PSM1_tool_main_link
              └─[roll,     revolute]─→ PSM1_tool_wrist_link
                └─[roll_shaft, fixed]─→ PSM1_tool_wrist_shaft_link
                  └─[wrist_pitch, revolute]─→ PSM1_tool_wrist_sca_link
                    └─[wrist_yaw, revolute]─→ PSM1_tool_wrist_sca_shaft_link
                      └─[jaw,   revolute]─→ PSM1_tool_wrist_sca_ee_link_0
                      └─[tool_tip, fixed]─→ PSM1_tool_tip_link
```

### 5.2 PSM 基座的确定

`fixed` 关节定义了 PSM 基座在世界坐标系里的位置。URDF 里写的是 `xyz="-0.25 0.0 0.5" rpy="0.0 0.0 3.1416"`。

**但这是在变换前的世界坐标系（= 相机坐标系）里**。做完桌面变换后，PSM 基座也要跟着变换。

**如何确定 PSM 在相机坐标系中的真实位置**：通过 handeye.yaml。

```yaml
PSM1_rvec: [0.978, -2.412, 1.123]    # 末端在相机坐标系下的旋转
PSM1_tvec: [93.81, -56.44, 0.762]    # 末端在相机坐标系下的平移(mm)
```

结合 LND.json 的 DH 参数，反算 PSM 基座在相机坐标系中的位姿，然后应用 T。

### 5.3 PSM 关节的坐标系变换

每帧从 robots.json 读取 7 个关节角 `q`：

```
q[0] = outer_yaw 关节角     → 旋转 PSM1_outer_yaw_link
q[1] = outer_pitch 关节角    → 旋转 PSM1_outer_pitch_link
q[2] = outer_insertion 位移  → 平移 PSM1_tool_main_link (米!)
q[3] = outer_roll 关节角     → 旋转 PSM1_tool_wrist_link
q[4] = outer_wrist_pitch    → 旋转 PSM1_tool_wrist_sca_link
q[5] = outer_wrist_yaw      → 旋转 PSM1_tool_wrist_sca_shaft_link
q[6] = jaw 关节角            → 旋转夹爪
```

`q[2]` 是**平移关节**，单位是米。Warp 必须正确处理 prismatic 类型。

---

## 第六部分：坐标变换矩阵速查表

### 变换矩阵一览

| 名称 | 符号 | 用途 |
|------|------|------|
| 身体→世界 | `X_WB` | 物体局部坐标 → 世界坐标 |
| 相机→世界 | `X_WC` | 相机坐标 → 世界坐标 |
| 世界→相机 | `X_CW = inv(X_WC)` | 世界坐标 → 相机坐标（渲染用）|
| 相机内参 | `K` | 3D 相机坐标 → 2D 图像像素 |
| 桌面变换 | `T` (X_table_from_source) | 相机坐标 → 桌面坐标 |
| 立体外参 | `[I \| -baseline,0,0]` | 左相机 → 右相机 |
| FK (正运动学) | `X_WL(q)` | 关节角 q → 每个连杆的世界位姿 |

### 分步变换示例

以组织上的一个 3D 点为例，展示它在各种坐标系之间的转换：

```
步骤 0: 读数据
        像素 (u,v)=(800,500), Z=depth=0.090m

步骤 1: 2D→3D 反投影 (图像坐标 → 相机坐标)
        X = (800-860.4)×0.090/1742.8 = -0.0033m
        Y = (500-682.3)×0.090/1742.8 = -0.0094m
        Z = 0.090m
        p_cam = (-0.0033, -0.0094, 0.090)  ← 相机坐标系

步骤 2: 相机坐标 → 世界坐标 (变换前 X_WC=I)
        p_world = X_WC × p_cam = I × p_cam = (-0.0033, -0.0094, 0.090)

步骤 3: 世界坐标 → 身体坐标 (X_WB=I, 所以一样)
        p_body = inv(X_WB) × p_world = p_world

步骤 4: 世界坐标 → 相机坐标 → 像素投影
        p_cam2 = X_CW × p_world = inv(X_WC) × p_world = p_world (因为X_WC=I)
        u = fx·(X/Z) + cx = fx·(-0.0033/0.090) + 860.4 = fx·(-0.0367) + 860.4 ≈ 797
        v = fy·(Y/Z) + cy = fy·(-0.0094/0.090) + 682.3 = fy·(-0.1044) + 682.3 ≈ 500
        → 投影回接近 (800,500)，验证正确 ✅

步骤 5: 如果做桌面变换
        p_table = T × p_world
        → 新世界坐标，在桌面上方
```

---

## 第七部分：Demo 和 Super 的坐标系对比

| 项目 | PushT Demo | Super 数据集 (变换前) | Super 数据集 (变换后) |
|------|-----------|---------------------|---------------------|
| 世界原点 | 桌面中心 | 左相机光心 | 桌面中心 |
| Z 轴方向 | 朝上 | 朝前（深入场景） | 朝上 |
| 桌面碰撞面 | `z=0` 水平面 | `0.16x+0.58y+0.80z-0.09=0` 斜面 | `z=0` 水平面 |
| 重力 | (0,0,-9.8) | (0,0,-9.8) | (0,0,-9.8) |
| 重力⊥桌面 | ✅ | ❌ 差 37.2° | ✅ |
| 组织 X_WB | identity | identity | T |
| 地面 X_WB | identity | identity | T |
| 左相机 X_WC | 代码写死 | I | T |
| 右相机 X_WC | 代码写死 | I+基线 | T×(I+基线) |
| PSM 基座 | URDF 写死 | 待 handeye | T×handeye |
| 机械臂 | Panda (7 revolute) | PSM1 (6 revolute + 1 prismatic) | 同左 |
