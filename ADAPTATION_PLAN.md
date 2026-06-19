# 适配计划 v3 — 最终版

> 最后更新：2026-06-19

---

## 总体进度

| # | 阶段 | 状态 | 关键产出 |
|---|------|------|---------|
| 1 | 原始数据提取 | ✅ | robots.json, 1644对 rectified PNG, calib |
| 2 | 深度估计 | ✅ | depth/000000-depth.npy (1920×1080) |
| 3 | SAM2 分割 | ✅ | masks/000000-tissue.png, ground.png |
| 4 | 构建场景物体 | ✅ | tissue.json(1991点), ground.json(2000点), ground_plane.json |
| 5 | PSM URDF 适配 | ✅/待投影对齐 | 已从 dVRK xacro 重建完整 PSM，mimic 派生驱动，夹爪 visual/collision 保留 |
| 6 | 组装离线数据集 | ✅ | robots.json, cameras.json, stereo_left/right.mp4 |
| 7 | **坐标系变换** | ✅ | 右手系 table frame, ground_plane=[0,0,1,0], 左右目投影验证 |
| 8 | **PSM 基座位姿** | ✅ | 已用 fin-local LND/handeye/robots.json 修正 dVRK xacro URDF fixed base；左右目 mesh/keypoints 验证图已重生成 |
| 9 | **super_embodied.py** | ⏳ | 待做 |
| 10 | **demo 入口 + 测试** | ⏳ | 待做 |

---

## 坐标系变换结果

ground_plane z=0 水平面。所有坐标在桌面坐标系（right_handed_table_world_z_up_m）。

| 组件 | Z 范围 | 说明 |
|------|--------|------|
| 桌面点 | z=0 | 在平面上 |
| 组织 | z: 0~45mm | 桌面上方, 1991点 |
| 左相机 | z=89mm | 桌面上方, pos=(20.3,-78.7,89.0)mm |
| 右相机 | z=88mm | pos=(25.5,-78.4,88.2)mm, baseline=+5.306mm along rectified +X |

**验证图片**（保留在 `data/super/grasp5_native/`）：
- `scene_3d_right_handed.png` — 右手系 3D 场景三视角
- `projection_right_handed.png` — 左右相机 2D 投影
- `projection_overlay_right_handed.png` — 组织(红色)+桌面(蓝色)叠加到左右 RGB 图像

---


## Phase 5B: PSM URDF 完整重建 + mimic 派生驱动

### 当前结论

当前 `data/super/psm_robot/psm.urdf` 已经由 `scripts/rebuild_super_psm_from_dvrk_xacro.py` 从 dVRK Classic PSM1 xacro 重新生成。旧的手工适配 URDF、旧 mesh、旧 `psm_assembled.ply`、旧 dense PSM 点云已经清理；新的 URDF 保留完整 link/joint 拓扑和夹爪 visual/collision，并将 mimic 关系导出到 `psm_mimic_map.json`。

### 原始 grasp5 关节数据检查

原始数据文件：

```text
data/super/grasp5/grasp5.bag
```

bag 中只有 4 个 topic：

```text
/stereo/viewer/left/image
/stereo/slave/left/image
/stereo/slave/right/image
/dvrk/PSM1/slave/state_joint_current
```

唯一的关节 topic 是：

```text
/dvrk/PSM1/slave/state_joint_current
```

该 topic 类型为 `sensor_msgs/JointState`，共有 5458 条消息。第一条和全部消息的 joint name 集合一致，只有 7 个关节：

```text
outer_yaw
outer_pitch
outer_insertion
outer_roll
outer_wrist_pitch
outer_wrist_yaw
jaw
```

第一帧 position 长度为 7：

```text
[0.8408666008, -0.3093572335, 0.1309628824,
 2.8721231559, 0.0267807540, 0.5821763742,
 0.9998290288]
```

关节范围：

```text
outer_yaw:          0.755190602 .. 0.840866601
outer_pitch:       -0.309364956 .. -0.273284348
outer_insertion:    0.129110484 .. 0.151839633
outer_roll:         2.870655326 .. 3.074054571
outer_wrist_pitch:  0.017762745 .. 0.055611055
outer_wrist_yaw:    0.424091150 .. 0.582176374
jaw:               -0.483825124 .. 1.001952502
```

结论：

```text
grasp5 原始数据集没有单独记录 jaw_mimic_1 / jaw_mimic_2。
它只记录了一个 jaw 值。
夹爪两片的运动必须从 jaw 派生出来。
```

### 为什么仍然需要 mimic

dVRK xacro 里除了 7 个独立关节，还有 mimic 关节。mimic 的意思是“这个关节不单独给数据，而是跟随另一个关节运动”。

PSM base 并行机构中：

```text
pitch_1 =  pitch
pitch_2 =  pitch
pitch_3 = -pitch
pitch_4 = -pitch
pitch_5 =  pitch
```

SCA 工具夹爪中：

```text
jaw_mimic_1 =  0.5 * jaw
jaw_mimic_2 = -0.5 * jaw
```

这些 mimic 关节没有独立传感器数据，但它们对视觉和接触很重要。特别是夹爪：

```text
如果粗暴删除 jaw_mimic_1 / jaw_mimic_2，
夹爪两片的 mesh/collision 就可能消失或不随 jaw 正确运动，
组织接触到夹爪时就不可信。
```

因此最终目标不是删除 mimic 的运动效果，而是把 mimic 从“额外输入关节”转换为“由 7 个输入 q 自动计算出的派生关节”。

### 完整重建路线

#### 1. 从 dVRK xacro 展开完整 PSM

输入：

```text
data/super/dvrk_model/urdf/Classic/PSM1.urdf.xacro
data/super/dvrk_model/urdf/Classic/psm.urdf.xacro
data/super/dvrk_model/urdf/Classic/psm_base.urdf.xacro
data/super/dvrk_model/urdf/Classic/psm_tool_sca.urdf.xacro
data/super/dvrk_model/meshes/Classic/PSM/*.dae
```

输出到：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/meshes/*.stl
```

要求：

- 保留 dVRK 原始 link / joint 拓扑；
- 保留 visual mesh；
- 为接触需要保留或补全 collision；
- DAE 转 STL；
- package mesh 路径改成当前 repo 相对路径；
- 写入右手系 table world 下的 `world -> PSM1_psm_base_link` fixed joint。

#### 2. 保留 7 个独立输入关节

`robots.json` 和原始 bag 只有 7 个独立关节，所以外部输入仍然保持 7 维：

```text
outer_yaw
outer_pitch
outer_insertion
outer_roll
outer_wrist_pitch
outer_wrist_yaw
jaw
```

不要要求数据集提供 `jaw_mimic_1`、`jaw_mimic_2`、`pitch_1` 等额外 q，因为原始数据里没有。

#### 3. 生成 mimic_map

新增一个显式映射文件，例如：

```text
data/super/psm_robot/psm_mimic_map.json
```

内容记录：

```json
{
  "pitch_1": {"source": "pitch", "multiplier": 1.0, "offset": 0.0},
  "pitch_2": {"source": "pitch", "multiplier": 1.0, "offset": 0.0},
  "pitch_3": {"source": "pitch", "multiplier": -1.0, "offset": 0.0},
  "pitch_4": {"source": "pitch", "multiplier": -1.0, "offset": 0.0},
  "pitch_5": {"source": "pitch", "multiplier": 1.0, "offset": 0.0},
  "jaw_mimic_1": {"source": "jaw", "multiplier": 0.5, "offset": 0.0},
  "jaw_mimic_2": {"source": "jaw", "multiplier": -0.5, "offset": 0.0}
}
```

#### 4. 在 FK / 验证脚本中展开 full q

验证脚本读取原始 7 维 q：

```text
q7 = robots["sheep"]["states"][frame]["q"]
```

然后生成完整关节字典：

```text
yaw         <- outer_yaw
pitch       <- outer_pitch
insertion   <- outer_insertion
roll        <- outer_roll
wrist_pitch <- outer_wrist_pitch
wrist_yaw   <- outer_wrist_yaw
jaw         <- jaw

pitch_1     <-  pitch
pitch_2     <-  pitch
pitch_3     <- -pitch
pitch_4     <- -pitch
pitch_5     <-  pitch
jaw_mimic_1 <-  0.5 * jaw
jaw_mimic_2 <- -0.5 * jaw
```

这样 `psm_frame0_right_handed_dense.ply` 会采样完整机器人，包括并行机构和两片夹爪。

#### 5. 在 Warp / embodied 环境中处理 mimic

如果 Warp 不能直接支持 URDF `<mimic>` 标签，则不要依赖 Warp 自动解析 mimic。

推荐做法：

1. URDF 中保留对应的 revolute joint 和 link/collision。
2. 可以去掉 `<mimic>` 标签，避免 Warp parser 不支持。
3. 另存 `psm_mimic_map.json`。
4. 每次从 7 维控制量/轨迹生成 full q。
5. full q 用于设置 articulation 状态，使夹爪 collision 跟随 jaw 运动。

这样做的含义是：

```text
数据集仍然是 7 个真实记录的关节；
仿真/渲染内部可以有更多从属关节；
从属关节不是新的传感器数据，而是由 7 个 q 算出来。
```

#### 6. 接触建模要求

为了让组织接触夹爪可信，重建后的 URDF 必须满足：

- `tool_wrist_sca_ee_link_1` 和 `tool_wrist_sca_ee_link_2` 必须存在；
- 两片夹爪必须有 visual mesh；
- 两片夹爪必须有 collision 几何，不能只有 visual；
- 两片夹爪必须随 `jaw` 派生运动，而不是固定不动；
- 验证时要检查 jaw 不同角度下两片夹爪是否开合。

### 2026-06-18 重建结果

已执行：

```text
python scripts/rebuild_super_psm_from_dvrk_xacro.py
python scripts/generate_super_psm_validation.py --points 200000
```

新增/更新产物：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/meshes/*.stl
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm_rebuild_report.json
data/super/psm_robot/psm_frame0_right_handed_dense.ply
data/super/grasp5_native/psm_projection_right_handed_dense.png
data/super/grasp5_native/psm_scene_right_handed_dense.png
```

结构校验：

```text
link_count: 18
joint_count: 17
visual_link_count: 15
collision_link_count: 17
mimic_tags_in_urdf: 0
mimic_joint_count_in_json: 7
unreachable_links_from_world: []
```

200000 点 dense 采样实际输出 200390 点，已包含原来缺失的并行机构和两片夹爪：

```text
PSM1_outer_pitch_back_link
PSM1_outer_pitch_front_link
PSM1_outer_pitch_bottom_link
PSM1_outer_pitch_top_link
PSM1_outer_insertion_link
PSM1_tool_wrist_sca_ee_link_1
PSM1_tool_wrist_sca_ee_link_2
```

2026-06-19 已完成 URDF base correction：PSM mesh/keypoints 已重新投影验证，左右目 `mesh_visible=604/200390`、`keypoints_visible=21/24`。详细记录见本文件的 “2026-06-19 更新：URDF Base Correction 已完成”。

### 验证标准

重建完成后必须检查：

1. 原始 7 维 q 能展开为完整 full q。
2. URDF 从 `world` 能遍历到所有有 visual/collision 的 link。
3. `psm_frame0_right_handed_dense.ply` 不再缺 `psm_assembled.ply` 中的主要零件。
4. jaw 改变时，两片夹爪会按相反方向运动。
5. 两片夹爪有 collision，可以用于组织接触。
6. PSM 投影图重新生成，并继续排查 PSM 相机坐标空间对齐问题。

---


## Phase 5C: 在 fin 内重建 PSM/LND 中间数据

### 目标

不放弃 `grasp5/LND.json` 和 `grasp5/handeye.yaml`。它们仍然是 PSM 的原始数据来源。

本阶段要放弃的是：

```text
embodied_gaussians_fixed_super2 里的旧 PSM/LND 中间产物
```

原因是旧中间产物混合了 raw camera、raw_camera_tuned、rectified camera、table world 等多个空间，而且字段名有历史遗留。当前 fin 必须重新生成自己的 PSM/LND 中间数据，保证之后每一步都可追踪、可复现。

### 输入

```text
data/super/grasp5/LND.json
data/super/grasp5/handeye.yaml
data/super/grasp5/camera_calibration.yaml
data/super/grasp5_native/calib_rectified.json
data/super/grasp5_native/timestamps.json
data/super/grasp5_offline_demo/robots.json
data/super/grasp5_offline_demo/super_ground_z_axis_realign.json
```

其中：

- `LND.json`：PSM 的 DH 运动学、关键点、骨架线定义。
- `handeye.yaml`：PSM 和 raw left camera 的 hand-eye 外参。
- `camera_calibration.yaml`：原始双目内外参，可重新计算 R1/R2。
- `robots.json`：5458 条 7 维 PSM joint 数据。
- `super_ground_z_axis_realign.json`：rectified camera 到 right-handed table world 的变换。

### 输出

输出目录：

```text
data/super/grasp5_offline_demo/instruments/
```

目标产物：

```text
psm1_lnd_model.json
psm1_lnd_motion.json
psm1_lnd_generation_report.json
psm1_lnd_projection_debug/
```

同时补齐：

```text
data/super/grasp5_native/calib_rectified.json
```

需要在其中保留：

```text
R1
R2
Q
original_size / rectified_size
K_left_rect / K_rectified
```

### 需要同时保存的坐标空间

为了避免再次混淆，motion 中必须显式保存多个字段，而不是只叫一个笼统的 `rectified_camera`：

```text
keypoints_psm_base
keypoints_raw_camera
keypoints_raw_camera_tuned
keypoints_rectified_from_raw_camera
keypoints_rectified_from_raw_camera_tuned
keypoints_projection_space
keypoints_table_from_rectified_raw
keypoints_table_from_projection_space
```

link transform 也同理保存：

```text
link_transforms_psm_base
link_transforms_raw_camera
link_transforms_raw_camera_tuned
link_transforms_rectified_from_raw_camera
link_transforms_rectified_from_raw_camera_tuned
link_transforms_table_from_rectified_raw
link_transforms_table_from_projection_space
```

### 候选投影诊断

第 0 帧必须输出候选投影统计，至少比较：

```text
raw_camera
default raw_camera_tuned / projection_space
rectified_from_raw_camera
rectified_from_raw_camera_tuned
table_from_rectified_raw, 再投回相机
table_from_projection_space, 再投回相机
```

当前已知现象：

```text
raw_camera_tuned / projection_space 约 21/24 keypoints 可见
rectified_camera 约 0/24 keypoints 可见
```

新脚本必须在 fin 内重新验证这个现象，不能再读 super2。

### 判断标准

本阶段成功的标准不是立刻修好最终 URDF base，而是先获得一份自洽诊断数据：

1. 所有 PSM/LND 中间数据都在 `embodied_gaussians_fixed_super_fin` 内生成。
2. 不再读取 `embodied_gaussians_fixed_super2` 的 PSM motion/model。
3. `calib_rectified.json` 补齐 R1/R2。
4. `psm1_lnd_generation_report.json` 明确说明哪个空间 keypoints 可见、哪个不可见。
5. 下一步再基于这个结果决定：
   - 走 raw_camera_tuned -> table world 的临时可视对齐；或
   - 修正 raw -> rectified -> table world 的严格坐标链。

### 2026-06-18 生成结果

已执行：

```text
python scripts/build_super_psm_lnd_intermediates.py
python scripts/generate_super_psm_validation.py --points 200000
```

生成产物：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_projection_debug/*.png
```

同时补齐：

```text
data/super/grasp5_native/calib_rectified.json: R1, R2, Q
```

第 0 帧 fin-local 诊断：

```text
raw_camera:                         21/24 keypoints in image
raw_camera_tuned:                   21/24 keypoints in image
projection_space:                   21/24 keypoints in image
rectified_from_raw_camera:          21/24 keypoints in image
rectified_from_raw_camera_tuned:    21/24 keypoints in image
table_from_projection_space 回投影: 21/24 keypoints in image
```

PSM 专项验证更新：

```text
keypoints_visible: 21/24
mesh_visible: 0/200390
```

这说明 `LND + handeye + joints` 在 fin 内可以重新投回图像；当前剩余问题集中在“dVRK xacro URDF mesh 的 base 坐标系”与“LND/handeye base 坐标系”没有对齐。

---

## Phase 8: PSM 基座位姿

### 目标

确定 PSM 机械臂基座在**桌面坐标系（right_handed_table_world_z_up_m）**中的位姿，写入 URDF 的 `fixed` 关节。

### 已知数据

| 数据 | 来源 | 含义 |
|------|------|------|
| `handeye.yaml` | grasp5 原始 | PSM 末端在相机坐标系下的位姿 (rvec, tvec) |
| `LND.json` | grasp5 原始 | PSM 器械的 DH 参数和 link 结构 |
| `robots.json` | Phase 1 | 所有帧的 7 个关节角 |
| `psm.urdf` | Phase 5 | PSM 运动学链，fixed 关节在 xyz="-0.25 0.0 0.5" |
| 桌面变换 X | Phase 7 | 相机坐标系 → 桌面坐标系的变换矩阵 |

### 当前信息来源：fin-local PSM/LND 重建

当前不再依赖 super2 的 PSM/LND 中间产物。fin 已经生成：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
```

第 0 帧 LND keypoints 已能投回左右目图像：

```text
keypoints_visible = 21 / 24
```

但完整 dVRK xacro URDF dense mesh 仍然：

```text
mesh_visible = 0 / 200390
```

因此 Phase 8 下一步不再是“从 super2 提取 base”，而是要对齐两个 base：

```text
LND/handeye base  -> 已经能投影 keypoints
dVRK xacro URDF base -> dense mesh 仍不在图像内
```

### 步骤

#### 第1步：读取 fin-local PSM/LND 模型数据

使用 `data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json` 中的 handeye/LND 变换矩阵，以及 `psm1_lnd_motion.json` 的第 0 帧 keypoints/link transforms。不要再读 super2。

#### 第2步：相机坐标系 → 桌面坐标系

PSM 基座在相机坐标系下的位姿 `X_CB`（Camera→Base），乘桌面变换 X：
```
X_TB = Rx(180) @ X @ X_CB
```
其中 X 是 Phase 7 的对齐矩阵，Rx(180)=diag(1,-1,-1,1) 用于保持右手系且让相机位于桌面上方。

#### 第3步：更新 URDF 的 fixed 关节

URDF 中 `fixed` 关节定义基座在世界坐标系（= 桌面坐标系）中的位置：
```xml
<joint name="fixed" type="fixed">
  <parent link="world"/>
  <child link="PSM1_psm_base_link"/>
  <origin rpy="..." xyz="..."/>   ← 从 X_TB 提取 rpy 和 xyz
</joint>
```

### 验证方法

1. 更新 URDF 后，用 `add_articulation_from_urdf` 加载，检查 PSM body 0 的初始位姿
2. 把 PSM mesh 投影到图像上，和 SAM2 instrument mask 对比
3. LND keypoints 当前已有 21/24 落在图像内；下一步需要让 dVRK xacro URDF mesh 也落到同一区域

---

## Phase 9: super_embodied.py

仿照 `pusht_embodied.py`，核心差异：

| 项目 | pusht_embodied | super_embodied |
|------|---------------|----------------|
| Q_START | Panda 7 角 | PSM1 7 角 (从 robots.json 第一帧) |
| URDF | panda.urdf | psm.urdf (含正确的基座位姿) |
| 可动物体 | tblock | tissue |
| 视觉物体 | ground | ground |
| 地面 | [0,0,1,0] | [0,0,1,0] (同样) |
| 静态相机 | add_static_cameras() | 不需要 |
| 高斯范围 | 全部 link | **仅尖端 link** |

### 仅尖端高斯

在 `add_renderable_articulation_from_urdf` 后，手动删除近端 link 的 gaussians：
- **保留高斯的 link**：tool_wrist_sca_link, tool_wrist_sca_shaft_link, tool_wrist_sca_ee_link_1, tool_wrist_sca_ee_link_2
- **删除高斯的 link**：psm_base, outer_yaw, outer_pitch, tool_main, tool_wrist, tool_wrist_shaft

方式：在 `gaussian_body_ids` 中把不需要渲染的 link 的 body_id 设为 -1，或修改 builder 只对指定 link 采样。

---

## Phase 10: Demo 入口 + 测试

- `examples/example_embodied_super_offline.py` — 复制 `example_embodied_pusht_offline.py`，改 import + 默认数据集路径
- 双相机：DatasetManager 自动读 cameras.json 里的两个相机
- FPS 设为 30（匹配视频帧率）
- 运行测试 → 修 bug

---

## 文件清单

```
最终数据文件：
  data/super/grasp5_offline_demo/
    robots.json, cameras.json
    videos/stereo_left.mp4, stereo_left.json
    videos/stereo_right.mp4, stereo_right.json

  examples/embodied_environments/super_embodied/objects/
    tissue.json (1991点, z≥0)
    ground.json (2000点, z=0)

  examples/embodied_environments/super_embodied/environment/
    ground_plane.json [0,0,1,0]

  data/super/psm_robot/
    psm.urdf, meshes/*.stl, psm_mimic_map.json, psm_rebuild_report.json, psm_frame0_right_handed_dense.ply

验证图片：
  data/super/grasp5_native/
    scene_3d_right_handed.png, projection_right_handed.png, projection_overlay_right_handed.png

注意：旧的 psm_keypoints/psm_frame0/psm_assembled 产物已清理；2026-06-19 已完成 URDF base correction，并重生成当前 PSM dense 点云和左右目投影验证图。
```


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
