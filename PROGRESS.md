# Embodied Gaussians — grasp5 外科手术数据集适配进度

> 最后更新：2026-06-19
>
> 目标：让 dVRK PSM1 手术器械操作软组织的离线数据，在 Embodied Gaussians Demo 中跑起来。
>
> 全部工作在 `embodied_gaussians_fixed_super_fin` 下完成。

---

## 总体进度

| # | 阶段 | 状态 | 关键产出 |
|---|------|------|---------|
| 1 | 原始数据提取 | ✅ | robots.json(5458关节步), 1644对 rectified PNG, calib |
| 2 | 深度估计 | ✅ | depth/000000-depth.npy (1920×1080, 中位95mm) |
| 3 | SAM2 分割 | ✅ | masks/000000-tissue.png, ground.png |
| 4 | 构建场景物体 | ✅ | tissue.json(1991点), ground.json(2000点), ground_plane.json |
| 5 | PSM URDF 适配 | ✅/待投影对齐 | 已从 dVRK xacro 重建完整 PSM，mimic 派生驱动，夹爪 visual/collision 保留 |
| 6 | 组装离线数据集 | ✅ | robots.json, cameras.json, stereo_left/right.mp4 |
| 7 | 坐标系变换 | ✅ | 右手系 table frame, ground_plane=[0,0,1,0], 左右目 overlay 验证 |
| 8 | PSM 基座位姿 | ✅ | 已用 fin-local LND/handeye/robots.json 修正 dVRK xacro URDF fixed base；左右目 mesh/keypoints 验证图已重生成 |
| 9 | super_embodied.py | ⏳ | 待做 |
| 10 | demo 入口 + 测试 | ⏳ | 待做 |

---

## Phase 1: 原始数据提取

### 脚本

`scripts/extract_grasp5_bag.py`

### 过程

从 `data/super/grasp5/grasp5.bag` (18GB, 4个topic) 提取：

| Topic | 消息数 | 产出 |
|-------|--------|------|
| `/dvrk/PSM1/slave/state_joint_current` | 5458 | robots.json (7关节: 6 revolute + 1 prismatic) |
| `/stereo/slave/left/image` | 1644 | 1644张 rectified PNG (1920×1080) |
| `/stereo/slave/right/image` | 1644 | 1644张 rectified PNG |
| `/stereo/viewer/left/image` | 1410 | 未使用 |

**遇到的坑**：
- 第1441帧编码是YUV但标记为rgb8 → 脚本加bpp自动检测解决
- camera_calibration.yaml的T向量单位是毫米 → baseline=5.3mm（不是5.3m）
- `rosbags`库需要手动解析二进制消息（不用内置的反序列化）

### 产出文件

```
data/super/grasp5_native/
├── robots.json              (2.7MB, 5458 关节步)
├── calib_rectified.json     (K=1742.8, baseline=0.0053m)
├── timestamps_left.json     (1644 条目)
├── timestamps_right.json    (1644 条目)
└── rgb/
    ├── 000000-left.png ~ 001643-left.png   (1920×1080)
    └── 000000-right.png ~ 001643-right.png
```

---

## Phase 2: 深度估计

从 super2 拿 Python-SuPer 手术专用管线生成的深度图（我们尝试了 OpenCV SGBM 和 RAFT-Stereo 通用权重，效果都差）：

- `data/super/grasp5_native/depth/000000-depth.npy`
- 原始 960×540 → 上采样到 1920×1080
- 100% 覆盖率，中位深度 95mm

---

## Phase 3: SAM2 图像分割

使用 super2 的 SAM2 mask（`grasp5_processed_before_z_axis_realign/sam2_masks/`），540p 分辨率。该 mask 由 SAM2 automatic mask generator 生成。

也写了一个交互式标注脚本 `scripts/sam2_segment.py`，通过 VNC `:99` / 端口 `:5912` 用鼠标点击做分割（使用 SAM2ImagePredictor 的 point-click 模式）。

产出：
```
data/super/grasp5_native/masks/
├── 000000-tissue.png   (960×540, binary)
└── 000000-ground.png   (960×540, binary)
```

---

## Phase 4: 构建场景物体

### 使用脚本

super2 的 `scripts/super/build_super_tissue.py` 和 `build_super_ground.py`。

这两个脚本：
1. SAM2 mask + 深度图 → 反投影到 3D 点云（相机坐标系，X_WB=I）
2. tissue: 多项式表面拟合填补深度缺失 → 降采样 → 2000个 particle + 2000个 gaussian
3. ground: 从深度点拟合平面 → 在平面上均匀采样 → 筛选 → 2000点
4. 拟合 ground plane 方程

运行命令（在 super2 目录下执行）：
```bash
cd embodied_gaussians_fixed_super2

python scripts/super/build_super_tissue.py \
  --native .../grasp5_native \
  --processed .../grasp5_offline_demo \
  --fixed-frame 0 \
  --mask-path .../masks/000000-tissue.png \
  --max-points 2000 --particle-radius 0.003 --gaussian-scale 0.0025 \
  --overwrite

python scripts/super/build_super_ground.py \
  --native .../grasp5_native \
  --processed .../grasp5_offline_demo \
  --frame 0 --fill-under-tissue \
  --tissue .../bodies/tissue.json \
  --ground-mask .../masks/000000-ground.png \
  --max-points 2000 --overwrite
```

### 产出（变换前，相机坐标系，X_WB=I）

```
examples/embodied_environments/super_embodied/objects/
├── tissue.json   (X_WB=I, 2000点, z:67-117mm)
└── ground.json   (X_WB=I, 2000点, z:88-165mm)

examples/embodied_environments/super_embodied/environment/
└── ground_plane.json  (0.156x+0.584y+0.797z-0.089=0, 法向量偏37.2°)
```

---

## Phase 5: PSM1 机械臂 URDF 适配

### 来源

jhu-dvrk/dvrk_model (GitHub: `https://github.com/jhu-dvrk/dvrk_model`)

### 适配过程

历史手工版本曾经做过：DAE → STL、删除 mimic、添加简化 collision/inertial。2026-06-18 复查发现该版本会漏掉部分装饰/夹爪 link，因此已改为从 dVRK xacro 可复现重建：

1. **xacro 展开**：从 `data/super/dvrk_model/urdf/Classic/PSM1.urdf.xacro` 展开完整 PSM1 + SCA tool。
2. **DAE → STL 转换**：将 dVRK Classic PSM mesh 转为 `data/super/psm_robot/meshes/*.stl`。
3. **保留 mimic 拓扑**：保留 `pitch_1`~`pitch_5`、`jaw_mimic_1/2` 的真实 joint/link，但把 `<mimic>` 标签导出为 `psm_mimic_map.json`。
4. **补全 contact 所需 collision**：对有 visual mesh 的 link 添加同 mesh collision；虚拟 link 添加很小的占位 collision。
5. **写入右手系基座位姿**：保留 Phase 8 修正后的 `world -> PSM1_psm_base_link` fixed origin。

### 2026-06-18 复查结论

原始 `data/super/grasp5/grasp5.bag` 中唯一的关节 topic 是 `/dvrk/PSM1/slave/state_joint_current`，共有 5458 条 `sensor_msgs/JointState`。全部消息的 joint name 集合一致，只有 7 个独立关节：

```text
outer_yaw, outer_pitch, outer_insertion, outer_roll,
outer_wrist_pitch, outer_wrist_yaw, jaw
```

原始数据没有单独记录 `jaw_mimic_1` / `jaw_mimic_2`，也没有单独记录 `pitch_1`~`pitch_5`。这些 dVRK mimic 关节必须从 7 个输入 q 派生出来，不能要求数据集额外提供。

当前 `psm.urdf` 已通过 `scripts/rebuild_super_psm_from_dvrk_xacro.py` 从 dVRK xacro 完整重建：保留 link/mesh/collision，保存 `psm_mimic_map.json`，运行时由 7 维 q 展开 full q，使夹爪两片随 `jaw` 运动并参与组织接触。旧的手工适配 URDF、旧 mesh、旧 `psm_assembled.ply` 和旧 PSM dense 点云已清理。

### 当前 URDF

7 个活动关节（匹配 robots.json 顺序）：

```
0. yaw         revolute
1. pitch       revolute
2. insertion   prismatic  ← 唯一平移关节, 值单位米
3. roll        revolute
4. wrist_pitch revolute
5. wrist_yaw   revolute
6. jaw         revolute
```

当前 xacro 重建结果：link_count=18, joint_count=17, visual_link_count=15, collision_link_count=17, URDF 内 mimic_tags=0，`psm_mimic_map.json` 中 mimic_joint_count=7，unreachable_links_from_world=[]。

### 产出

```
data/super/psm_robot/
├── psm.urdf                          (xacro 重建，18 link / 17 joint)
├── meshes/*.stl                      (13个 STL 文件，来自 dVRK Classic PSM)
├── psm_mimic_map.json                (7维 q -> mimic 派生 q)
├── psm_rebuild_report.json           (重建报告)
└── psm_frame0_right_handed_dense.ply (200390点，含并行机构和两片夹爪)
```

### 2026-06-18 重建验证

运行：

```text
python scripts/rebuild_super_psm_from_dvrk_xacro.py
python scripts/generate_super_psm_validation.py --points 200000
```

采样统计确认之前缺失的 link 已进入 dense 点云：

```text
PSM1_outer_pitch_back_link:      3465
PSM1_outer_pitch_front_link:     3619
PSM1_outer_pitch_bottom_link:   25109
PSM1_outer_pitch_top_link:       9616
PSM1_outer_insertion_link:      36175
PSM1_tool_wrist_sca_ee_link_1:    100
PSM1_tool_wrist_sca_ee_link_2:    100
```

第 0 帧 q 展开结果中，夹爪 mimic 已由 `jaw` 派生：

```text
jaw:          0.9998290288
jaw_mimic_1: 0.4999145144
jaw_mimic_2:-0.4999145144
```

2026-06-19 已完成 URDF base correction：PSM 投影图已重生成，左右目 `mesh_visible=604/200390`、`keypoints_visible=21/24`。这次修的是 URDF fixed base，不是 URDF link 缺失问题。

### Phase 5C: fin-local PSM/LND 中间数据

已新增脚本：

```text
scripts/build_super_psm_lnd_intermediates.py
```

已生成：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json          (9.4KB)
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json         (100MB, 1642帧)
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_projection_debug/*.png
```

同时补齐：

```text
data/super/grasp5_native/calib_rectified.json: R1, R2, Q
```

第 0 帧候选投影诊断：

```text
raw_camera:                         21/24 keypoints in image
raw_camera_tuned:                   21/24 keypoints in image
projection_space:                   21/24 keypoints in image
rectified_from_raw_camera:          21/24 keypoints in image
rectified_from_raw_camera_tuned:    21/24 keypoints in image
table_from_projection_space 回投影: 21/24 keypoints in image
```

`generate_super_psm_validation.py` 已改为读取 fin-local：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
```

不再读取 `embodied_gaussians_fixed_super2` 的 PSM motion/model。

重跑 PSM 验证后：

```text
keypoints_visible: 21/24
mesh_visible: 0/200390
```

结论：`LND + handeye + joints` 在 fin 内已经能投回图像；2026-06-19 已继续完成 dVRK xacro URDF mesh 的 base correction，使 URDF fixed base 与 LND/handeye/table-world 对齐。

---


## 2026-06-19 更新：URDF Base Correction 已完成

旧的 URDF fixed base 来自早期 super2/table-world 中间结果，不能继续作为 fin 当前坐标系的最终基座位姿。现在已经改为只使用 fin 内的当前数据重新拟合：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/robots.json
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm.urdf
```

新增脚本：

```text
scripts/fit_psm_urdf_base_to_lnd.py
```

它先把 URDF 的 `fixed` base 临时当作 identity，使用第 0 帧的 7 个输入关节和 mimic_map 做 FK，然后用稳定对应点做刚体拟合：

```text
URDF PSM1_outer_pitch_link          -> LND link0/base
URDF PSM1_tool_main_link            -> LND link3/insertion side
URDF PSM1_tool_wrist_link           -> LND link4/wrist
URDF PSM1_tool_wrist_sca_shaft_link -> LND distal/ee 平均点
URDF PSM1_tool_tip_link             -> LND distal/ee 平均点（低权重）
```

没有使用 `grip_far`，因为第 0 帧里它的 z 坐标在桌面 z=0 下方，直接用它会把夹爪尖端往错误方向拉。

当前已写入 `data/super/psm_robot/psm.urdf` 的 fixed origin：

```xml
<origin rpy="0.733342957 0.492181364 2.02256874" xyz="0.0994293715 -0.0484231708 0.0993431732" />
```

拟合报告：

```text
data/super/psm_robot/psm_base_correction_report.json
```

误差：

```text
mean residual: 0.00246 m
max residual:  0.00751 m
```

旧的 PSM 验证图/点云已删除，并重生成：

```text
data/super/grasp5_native/psm_projection_right_handed_dense.png
data/super/grasp5_native/psm_scene_right_handed_dense.png
data/super/psm_robot/psm_frame0_right_handed_dense.ply
```

第 0 帧投影统计：

```text
stereo_left:  mesh_visible=604/200390, keypoints_visible=21/24
stereo_right: mesh_visible=604/200390, keypoints_visible=21/24
```

说明：内窥镜视野主要看到器械末端，PSM base 和大部分远端机构在画面外，所以 mesh_visible 数量远小于总采样点数是正常的。

## Phase 6: 组装离线数据集

### 过程

1. 从 rectified PNG 序列生成 MP4（OpenCV VideoWriter, mp4v编码）
2. 生成每个相机的元数据 JSON（K, resolution, timestamps）
3. 生成 cameras.json（X_WC, video_path, metadata_path）
4. 复制 robots.json

### 产出

```
data/super/grasp5_offline_demo/
├── robots.json              (5458 关节步)
├── cameras.json             (stereo_left + stereo_right)
└── videos/
    ├── stereo_left.mp4      (130MB, 1644帧, 1920×1080)
    ├── stereo_left.json     (K, resolution, timestamps)
    ├── stereo_right.mp4     (160MB, 1644帧)
    └── stereo_right.json
```

---

## Phase 7: 坐标系变换

### 问题

原始数据在相机坐标系（世界原点 = 左相机光心）。桌面平面是斜的（法向量和Z轴差37.2°），重力不垂直于桌面。

### 方法

1. 从 ground 3D 点拟合平面方程：`0.156x + 0.584y + 0.797z - 0.089 = 0`
2. 用 scipy `Rotation.align_vectors` 计算旋转矩阵 R，将法向量对齐到 (0,0,1)
3. 平移 t = -R@p0（p0 是原点到平面的垂足），使桌面落在 z=0
4. 组合成变换 X = [R|t]，应用到所有世界坐标
5. 使用 Rx(180)=diag(1,-1,-1,1) 进行右手系旋转，确保相机和组织在 z>0（桌面上方）

### 变换矩阵

```
X = [[ 0.9865, -0.0507, -0.1559,  0.0203],
     [-0.0507,  0.8102, -0.5840,  0.0787],
     [ 0.1559,  0.5840,  0.7966, -0.0890],
     [ 0.0,     0.0,     0.0,     1.0   ]]

最终: 世界坐标 = Rx(180) @ X @ 相机坐标
     相机 OpenCV X_WC = Rx(180) @ X；cameras.json 存 Blender pose，FramesBuilder 内部转 OpenCV
```

### 变换后状态

| 组件 | 值 | 文件 |
|------|-----|------|
| ground_plane | [0,0,1,0] z=0水平面 | `super_embodied/environment/ground_plane.json` |
| tissue (1991点) | z: 0~45mm 桌面上方 | `super_embodied/objects/tissue.json` |
| ground (2000点) | z=0 | `super_embodied/objects/ground.json` |
| 左相机 | pos=(20.3,-78.7,89.0)mm | `grasp5_offline_demo/cameras.json` |
| 右相机 | pos=(25.5,-78.4,88.2)mm, baseline=+5.306mm along rectified +X | 同上 |

### 验证

- 左目组织投影：u[35,1655] v[207,1108]，93.7% 点在图像内
- 右目 baseline 符号已修正：右目相对左目 median du=-104.65px，93.3% 点在图像内
- 桌面点距离平面 < 10⁻¹⁶m
- 投影到真实 RGB 图像验证通过（`projection_overlay_right_handed.png`）

### 验证图片

```
data/super/grasp5_native/
├── scene_3d_right_handed.png              (右手系3D场景: 斜视+XZ+YZ三视角)
├── projection_right_handed.png            (左右相机2D投影)
└── projection_overlay_right_handed.png    (组织红色+桌面蓝色叠加到左右RGB图像)
```

---

## Phase 8: PSM 基座位姿 ✅

### 数据来源

super2 的 PSM 运动学模型文件：
- `grasp5_processed_before_z_axis_realign/.../instruments/psm1_lnd_model.json` — 含 `T_rectified_camera_psm_base`
- `grasp5_processed_before_z_axis_realign/.../instruments/psm1_lnd_motion.json` — 逐帧 link transforms + keypoints
- `grasp5_processed_before_z_axis_realign/.../instruments/urdf/psm1_urdf_articulation.json` — URDF 关节映射

### 计算过程

1. 读取 `T_rectified_camera_psm_base` — 基座在 rectified 相机坐标系中：(99.0, 27.1, -102.1)mm
2. 应用桌面变换：`T_our = Rx(180) @ X @ T_rectified_camera_psm_base`
   - X：Phase 7 的 4×4 旋转+平移矩阵
   - Rx(180)：右手系旋转 diag(1,-1,-1,1)
3. 从 T_our 提取 rpy（用 scipy Rotation.from_matrix）和 xyz

### 最终 URDF 参数

```xml
<joint name="fixed" type="fixed">
  <parent link="world"/>
  <child link="PSM1_psm_base_link"/>
  <origin rpy="-0.238342 -0.478597 2.466444" xyz="0.132451 -0.155286 0.139112"/>
</joint>
```

基座在右手系桌面坐标系：(132, -155, 139)mm，z>0 桌面上方。

### 验证

PSM 基座已用 `Rx(180) @ X @ T_rectified_camera_psm_base` 重算并写入 URDF。旧的 `psm_keypoints_projection.png`、`psm_projection_v2.png`、`psm_frame0.ply` 均来自右手系修正前，已不作为当前验证依据。

下一步需要重生成 PSM mesh/keypoints 投影图，确认器械尖端在左右目图像中的位置。

### 产出

- 更新 `data/super/psm_robot/psm.urdf` fixed joint 的 rpy 和 xyz
- `data/super/grasp5_native/psm_keypoints_projection.png` 是修正前验证图，右手系修正后需重生成
- `data/super/psm_robot/psm_frame0.ply` 是修正前验证点云，右手系修正后需重生成
  - 基座: (132,-155,139)mm；注意旧 psm_frame0.ply 是修正前验证产物，需重生成后再使用
  - 部分 link 在 z<0 但不影响物理仿真（ground plane 约束会处理）

## Phase 9-10: 待完成

详见 `ADAPTATION_PLAN.md`。

### Phase 9: super_embodied.py

- PSM URDF + tissue(rigid body) + ground(visual body) + ground_plane
- 仅尖端 link(wrist_sca, wrist_sca_shaft, jaw links) 放高斯

### Phase 10: demo 入口 + 测试

---

## 文件结构总览

```
embodied_gaussians_fixed_super_fin/
│
├── ADAPTATION_PLAN.md                     ← 适配计划
├── PROGRESS.md                            ← 本文件
├── COORDINATE_SYSTEMS.md                  ← 坐标系说明
│
├── scripts/
│   ├── extract_grasp5_bag.py              ← Phase 1: bag提取
│   ├── sam2_segment.py                    ← Phase 3: SAM2交互分割
│   └── realign_super_ground_to_z_axis.py  ← Phase 7: 桌面变换(从super2复制)
│
├── data/super/
│   ├── grasp5/                            ← 原始bag+标定(只读)
│   ├── grasp5_native/                     ← 提取+处理的中间数据
│   │   ├── rgb/                           ← 1644对 rectified PNG
│   │   ├── depth/                         ← 深度图
│   │   ├── masks/                         ← SAM2分割结果
│   │   ├── robots.json                    ← 关节数据
│   │   ├── calib_rectified.json           ← 标定
│   │   └── timestamps_*.json              ← 时间戳
│   ├── grasp5_offline_demo/               ← Demo直接读取的成品
│   │   ├── robots.json
│   │   ├── cameras.json
│   │   └── videos/stereo_{left,right}.mp4 + .json
│   ├── dvrk_model/                        ← jhu-dvrk官方仓库(只读)
│   └── psm_robot/                         ← 适配后的URDF+mesh
│       ├── psm.urdf
│       ├── meshes/*.stl
│       └── psm_assembled.ply
│
├── examples/embodied_environments/super_embodied/
│   ├── objects/
│   │   ├── tissue.json                    ← 1991点, z≥0, X_WB=I
│   │   └── ground.json                    ← 2000点, z=0, X_WB=I
│   └── environment/
│       └── ground_plane.json              ← [0,0,1,0]
│
└── third_party/
    └── (未使用，Python-SuPer直接从super2引用)
```
