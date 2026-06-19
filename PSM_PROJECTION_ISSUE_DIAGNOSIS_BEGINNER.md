# PSM 投影问题诊断说明（给初学者）

日期：2026-06-18

本文解释当前为什么 `psm_frame0_right_handed_dense.ply` 已经有完整 PSM 机器人，但投影到左右目图像时仍然看不到，也就是验证脚本输出：

## 2026-06-18 更新：fin-local LND 已重建

本文最初指出：旧验证脚本混用了 super2 的 PSM/LND 中间数据。现在这个问题已经处理：

```text
已在 embodied_gaussians_fixed_super_fin 内重新生成 PSM/LND 中间数据。
验证脚本已经改为读取 fin-local psm1_lnd_motion.json。
不再读取 embodied_gaussians_fixed_super2 的 PSM motion/model。
```

新生成文件：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_projection_debug/*.png
```

新结果：

```text
fin-local LND keypoints: 左右目 21/24 可见
URDF dense mesh:         0/200390 可见
```

因此当前结论进一步收窄为：

```text
LND + handeye + joints 不是要放弃，它们已经能在 fin 内投回图像。
剩下的问题是 dVRK xacro URDF mesh 的 base 坐标系没有对齐 LND/handeye base。
```

---

```text
mesh_visible = 0
keypoints_visible = 0
```

结论先说：

```text
这次不是 URDF 少零件的问题。
URDF 结构已经修好。
现在的问题是 PSM 的位姿/关键点使用了错误或混用的坐标空间，导致 PSM 末端跑到相机后方或图像外面。
```

---

## 1. 当前哪些已经是正确的

### 1.1 PSM URDF 结构已经完整

我们已经从 dVRK xacro 重新生成了完整 PSM：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm_rebuild_report.json
data/super/psm_robot/psm_frame0_right_handed_dense.ply
```

结构检查结果：

```text
link_count: 18
joint_count: 17
visual_link_count: 15
collision_link_count: 17
URDF 内 mimic_tags: 0
psm_mimic_map.json 中 mimic_joint_count: 7
unreachable_links_from_world: []
```

这说明：

```text
机器人零件现在不是孤立的。
所有 link 都能从 world 遍历到。
两片夹爪也已经有 visual 和 collision。
```

### 1.2 dense 点云已经采到之前缺失的零件

重新生成 dense 点云后，之前缺失的 link 已经采样到了：

```text
PSM1_outer_pitch_back_link:      3465
PSM1_outer_pitch_front_link:     3619
PSM1_outer_pitch_bottom_link:   25109
PSM1_outer_pitch_top_link:       9616
PSM1_outer_insertion_link:      36175
PSM1_tool_wrist_sca_ee_link_1:    100
PSM1_tool_wrist_sca_ee_link_2:    100
```

所以现在的问题不是“点云里没有夹爪/关节”。

---

## 2. 什么叫“投影看不到”

相机投影可以简单理解成：

```text
三维点 -> 相机坐标 -> 像素坐标
```

对 OpenCV 相机来说：

```text
相机坐标 z > 0：点在相机前方，有可能被看见
相机坐标 z < 0：点在相机后方，不可能被看见
```

然后还要看像素坐标是否落在图像范围内：

```text
0 <= u < 1920
0 <= v < 1080
```

只有同时满足这两个条件，点才算 visible。

当前验证结果已经分成两部分：

```text
URDF dense mesh:   mesh_visible = 0
fin-local LND keypoints: keypoints_visible = 21 / 24
```

意思是：

```text
LND/handeye 的关键点已经能投回图像；
但 dVRK xacro URDF 生成的 dense mesh 仍没有任何点既在相机前方又落在图像范围内。
```

---

## 3. 当前 PSM 点云在哪里

当前 PSM dense 点云是：

```text
data/super/psm_robot/psm_frame0_right_handed_dense.ply
```

它在右手系 table world 中的范围是：

```text
x: 0.0197  ~ 0.7210 m
y: -0.4057 ~ 0.4839 m
z: 0.0776  ~ 0.3809 m
```

中心大约是：

```text
(0.370, 0.039, 0.229) m
```

相比之下，组织范围是：

```text
x: -0.0442 ~ 0.0445 m
y: -0.0511 ~ 0.0146 m
z: 0.0011  ~ 0.0448 m
```

桌面范围是：

```text
x: -0.0787 ~ 0.0677 m
y: -0.0425 ~ 0.0561 m
z: 0
```

这说明整条 PSM 机械臂很大，包含底座和外部机构，所以整体 bbox 大是正常的。真正应该和组织接触、投影到画面里的主要是末端/夹爪，而不是整条大臂。

---

## 4. 关键证据：PSM 末端在相机后方

左相机在 table world 中的位置是：

```text
left camera center = (0.0203, -0.0787, 0.0890) m
```

当前 PSM 末端几个关键 link 的世界坐标大概是：

```text
PSM1_tool_wrist_link:          (0.0366, -0.1001, 0.1063)
PSM1_tool_wrist_sca_shaft:     (0.0290, -0.0960, 0.1036)
PSM1_tool_wrist_sca_ee_link_1: (0.0290, -0.0960, 0.1036)
PSM1_tool_wrist_sca_ee_link_2: (0.0290, -0.0960, 0.1036)
```

这些位置看起来离相机很近，但投影时要看“相机坐标 z”。按当前相机姿态转换后：

```text
PSM1_tool_wrist_link          z: -0.0378 ~ -0.0244 m
PSM1_tool_wrist_sca_shaft     z: -0.0323 ~ -0.0253 m
PSM1_tool_wrist_sca_link      z: -0.0312 ~ -0.0210 m
PSM1_tool_wrist_sca_ee_link_1 z: -0.0257 ~ -0.0127 m
PSM1_tool_wrist_sca_ee_link_2 z: -0.0255 ~ -0.0157 m
```

这些 z 都是负数。

意思是：

```text
按当前坐标链，夹爪/末端在相机后方 1~4 cm 左右。
所以相机不可能看到它们。
```

这就是 `mesh_visible=0` 的直接原因。

---

## 5. 为什么整机有些点 z>0 但还是看不到

整体 PSM 点云中，大约 35% 的点在左相机前方：

```text
depth positive count: 70716 / 200390
```

但这些点主要来自大臂/底座/外部机构，而且投影像素远远跑出图像范围：

```text
深度为正的点，像素范围大概：
u 最小约 6612，最大约 3.9e8
v 最大仍是 -3226，很多是巨大负数
```

图像范围应该是：

```text
u: 0 ~ 1919
v: 0 ~ 1079
```

所以即使有一部分大臂点在相机前方，它们也不在画面里。

真正应该出现在画面里的末端/夹爪反而在相机后方。

---

## 6. 另一个关键证据：旧 LND 数据的“可见空间”不是 rectified_camera

历史验证脚本曾经读取：

```python
motion_path = Path("/home/hsieh/data0/wad/embodied_gaussians_fixed_super2/.../psm1_lnd_motion.json")
key_source = motion["states"][0]["keypoints_rectified_camera"]
```

这个问题现在已经修复。

### 6.1 当前 fin 已经有自己的 PSM/LND 中间数据

当前验证脚本读取：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
```

这说明：

```text
PSM/LND keypoints 不再来自 super2。
URDF、LND motion、projection debug 都在 fin 内生成。
```

### 6.2 super2 的 LND 默认可视空间是 raw_camera_tuned

super2 的 `psm1_lnd_model.json` 里写了：

```text
default_motion_space: raw_camera_tuned
recommended_visual_space: raw_camera_tuned for image overlay
```

我检查了第 0 帧 keypoints，结果是：

```text
keypoints_raw_camera:        21 / 24 个点直接可见
keypoints_raw_camera_tuned:  21 / 24 个点直接可见
keypoints_projection_space:  21 / 24 个点直接可见
keypoints_rectified_camera:   0 / 24 个点直接可见
```

这非常重要。

它说明：

```text
旧 LND/handeye 调出来能投到图像上的空间，是 raw_camera_tuned / projection_space。
不是当前验证脚本拿来用的 rectified_camera。
```

所以当前脚本用 `keypoints_rectified_camera` 来验证 PSM，本身就很可疑。

---

## 7. raw / rectified / table world 是什么关系

简单理解：

```text
raw camera：原始左相机坐标系
rectified camera：立体校正后的左相机坐标系
table world：我们把桌面拉平后的世界坐标系
```

组织和桌面现在主要走的是：

```text
rectified camera -> right_handed_table_world_z_up_m
```

这个流程目前看起来是对的，因为 tissue/ground 左右目 overlay 已经吻合。

PSM 历史流程则比较复杂：

```text
handeye.yaml + LND.json + joints
-> raw_camera / raw_camera_tuned
-> 有时再尝试转 rectified_camera
-> 有时再转 table_world
```

当前问题是：

```text
组织/桌面采用 rectified/table 流程；
PSM 历史可见结果采用 raw_camera_tuned 流程；
我们现在把 PSM 当成 rectified/table 数据来用，结果坐标不对。
```

---

## 8. fin 里还缺一个重要校准信息

当前 fin 的：

```text
data/super/grasp5_native/calib_rectified.json
```

只保留了：

```text
K_rectified
P1
P2
baseline_m
```

但是没有保存：

```text
R1
R2
```

`R1` / `R2` 是 raw camera 到 rectified camera 的旋转信息。

这意味着：

```text
如果要在 fin 里重新、干净地从 handeye/LND 推出 PSM 的 rectified 位姿，当前校准文件信息不够完整。
```

虽然 `scripts/extract_grasp5_bag.py` 代码里会写 R1/R2，但当前实际的 `calib_rectified.json` 没有它们。这说明当前数据产物和脚本预期不一致。

---

## 9. 当前最可能的问题排序

### 问题 1：PSM 使用了错误的可视坐标空间

证据：

```text
raw_camera_tuned keypoints: 21/24 可见
rectified_camera keypoints: 0/24 可见
```

所以当前拿 rectified/table 路线直接放 PSM，很可能不对。

### 问题 2：当前 PSM base pose 不适合新 URDF/table frame

当前 PSM base 写在 URDF 的：

```xml
<joint name="fixed" type="fixed">
  <parent link="world"/>
  <child link="PSM1_psm_base_link"/>
  <origin rpy="-0.238342 -0.478597 2.466444" xyz="0.132451 -0.155286 0.139112"/>
</joint>
```

这个值来自之前 super2 的 rectified/table 中间结果。

但是现在检查发现，用它驱动完整 PSM xacro 后：

```text
夹爪/末端在左相机坐标里 z<0，也就是相机后方。
```

所以这个 base pose 至少对当前完整 URDF + 当前相机坐标链不成立。

### 问题 3：验证脚本不自洽

当前 `scripts/generate_super_psm_validation.py`：

```text
URDF 和 mesh 读 fin
keypoints 却读 super2
```

这会让诊断混乱。

下一步应该把 PSM/LND 中间数据在 fin 里重新生成，不要再依赖 super2 的旧文件。

### 问题 4：fin 缺 R1/R2，无法完全复现 raw→rectified

当前 fin 的校准文件缺少 R1/R2。

如果我们要从 `handeye.yaml` 干净地重建 PSM 到 rectified/table 的坐标链，就需要恢复或重新生成包含 R1/R2 的 `calib_rectified.json`。

---

## 10. 不是哪些问题

### 不是 URDF 缺零件

已经确认：

```text
unreachable_links_from_world = []
```

### 不是 mimic 没有实现

已经确认：

```text
jaw:          0.9998290288
jaw_mimic_1: 0.4999145144
jaw_mimic_2:-0.4999145144
```

夹爪两片已经会由 `jaw` 派生。

### 不是左右目 baseline 问题

tissue/ground 左右目 overlay 已经基本吻合。

PSM 左目本身都 `mesh_visible=0`，所以不是右目 baseline 单独导致的。

---

## 11. 建议下一步怎么修

我建议不要继续在当前 URDF fixed origin 上盲调。

应该分成三步。

### 第一步：让 fin 自己生成 PSM/LND 中间数据

在 `embodied_gaussians_fixed_super_fin` 内新增脚本，从这些输入重新生成 PSM 中间结果：

```text
data/super/grasp5/LND.json
data/super/grasp5/handeye.yaml
data/super/grasp5_offline_demo/robots.json 或 grasp5_native/robots.json
data/super/grasp5_native/calib_rectified.json
```

但在此之前要恢复 `calib_rectified.json` 中的：

```text
R1
R2
```

否则 raw→rectified 链不完整。

### 第二步：明确 PSM 到图像应该用哪个空间

目前证据显示：

```text
raw_camera_tuned / projection_space 可以投到图像上；
rectified_camera 不可以。
```

所以要做一个明确选择：

#### 方案 A：继续用 raw_camera_tuned 做 PSM 可视对齐

做法：

```text
handeye/LND -> raw_camera_tuned
raw_camera_tuned -> table world
```

难点：需要定义一个正确的 `X_table_world_from_raw_camera_tuned`。

#### 方案 B：修正 raw→rectified 链，让 PSM 真正进入 rectified camera

做法：

```text
handeye/LND -> raw camera
raw camera --R1/R2/rectification--> rectified camera
rectified camera -> table world
```

难点：要确认 handeye 方向、R1 使用方向、以及 tuned delta 是否还需要。

我更建议先做方案 B 的诊断，因为 tissue/ground/depth 已经建立在 rectified camera 上；长期看，PSM 也应该和它们统一到同一个 rectified/table world。

### 第三步：用图像 mask 反过来验证/优化 PSM base pose

不能只看 3D 数字，还要用图像验证。

具体做法：

1. 用 `LND.json` 的关键点和 dVRK xacro URDF 末端 link 对齐。
2. 用第 0 帧图像里的 instrument mask 或可见工具位置做目标。
3. 优化 `world -> PSM1_psm_base_link` 的 6D 位姿。
4. 检查左目和右目投影都落在工具区域内。
5. 再检查夹爪和组织接触关系。

---

## 12. 当前结论

当前 PSM 投影失败的直接原因是：

```text
按现在的 world/camera/base 坐标链，PSM 末端和夹爪在相机坐标里 z<0，已经跑到相机后方。
```

更深层原因是：

```text
PSM 历史可见数据在 raw_camera_tuned / projection_space；
当前 tissue/ground/camera 流程在 rectified/table world；
现在验证脚本混用了这些空间，并且还读取 super2 的旧中间数据。
```

所以下一步应该重建 fin 内部的 PSM 坐标链，而不是继续改 URDF 结构。

最重要的一句话：

```text
URDF 结构问题已经解决；现在剩下的是 PSM 位姿和相机坐标空间对齐问题。
```
