# super2 数据清理和上一步情况说明（给初学者）

日期：2026-06-18

本文只解释当前 `embodied_gaussians_fixed_super_fin` 里上一阶段发生了什么，以及为什么我刚才要清理一部分数据。

---

## 1. 先说结论

现在不是要放弃 `grasp5` 原始数据。

我们保留这些原始或可追踪数据：

```text
data/super/grasp5/LND.json
data/super/grasp5/handeye.yaml
data/super/grasp5/camera_calibration.yaml
data/super/grasp5/grasp5.bag
data/super/grasp5_native/rgb/
data/super/grasp5_native/robots.json
data/super/grasp5_native/calib_rectified.json
data/super/grasp5_offline_demo/robots.json
data/super/grasp5_offline_demo/cameras.json
data/super/grasp5_offline_demo/videos/
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
```

刚才清理的是明确来自 `embodied_gaussians_fixed_super2`，或者由 super2 数据继续派生出来的组织、桌面、深度、mask、旧验证图。

原因很简单：这些数据虽然之前能帮助我们快速验证左右目组织投影，但它们不是在当前 fin 目录中从原始数据完整重建出来的。你现在希望从 fin 自己的数据链继续往下做，所以这些旧来源数据应该先清掉，避免后面看图时误以为它们是 fin 自己生成的。

---

## 2. super2 和 fin 分别是什么

可以把它们理解成两个不同的工作目录：

```text
embodied_gaussians_fixed_super2
```

这是之前的旧工作目录。里面有一些已经跑出来的中间结果，比如深度图、SAM2 mask、旧的 PSM/LND motion。

```text
embodied_gaussians_fixed_super_fin
```

这是现在你要求继续整理的最终目录。我们应该尽量让它里面的数据都能从 `grasp5` 原始数据重新生成，不能偷偷依赖 super2。

之前为了快速验证，有些数据是从 super2 拿过来或用 super2 的脚本/中间结果生成的。这样短期能推进，但长期会有问题：

```text
你看见一个结果，但不知道它到底来自 fin 原始数据，还是来自 super2 旧中间结果。
```

所以现在清理 super2 数据是正确的。

---

## 3. 上一步我们做了什么

上一步主要做了两件事。

### 3.1 从 dVRK xacro 重建完整 PSM URDF

PSM 是手术机器人那条机械臂。

一开始的问题是：旧的 PSM 模型不完整，很多 link 是孤立的或者没有真实 joint 连起来。这样采样出来的点云会缺零件，也不能正确表示夹爪。

后来我们从 dVRK 的 xacro 文件重新展开 PSM，生成了：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/meshes/*.stl
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm_rebuild_report.json
```

这个结果的意义是：

```text
PSM 的结构基本完整了。
link 都能从 world 走到。
夹爪的左右两片也保留了 visual/collision。
mimic 关系没有直接写进 URDF，而是单独放在 psm_mimic_map.json 里。
```

这里的 mimic 可以理解成“跟着另一个关节一起动的关节”。比如夹爪只有一个 jaw 数值，但真实模型里左右夹爪要反方向张开，所以需要派生出两个 mimic jaw。

### 3.2 在 fin 内重新生成 PSM/LND 中间数据

`LND.json` 是原始数据里描述 PSM 运动学和关键点的文件。

`handeye.yaml` 是相机和 PSM 之间的外参关系。

`robots.json` 是每一帧的 7 个关节数据。

我们新写了脚本：

```text
scripts/build_super_psm_lnd_intermediates.py
```

它在 fin 内重新生成了：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_projection_debug/*.png
```

这一步很重要，因为它说明：

```text
PSM/LND 这条链现在不需要读 super2 的旧 motion/model。
```

---

## 4. 为什么 LND keypoints 可见，但 URDF mesh 不可见

这是上一轮最容易混淆的地方。

我们现在有两套和 PSM 有关的东西：

```text
LND keypoints
URDF dense mesh
```

它们不是同一个东西。

### 4.1 LND keypoints 是什么

LND keypoints 可以理解成“在机械臂上标出来的一些小点”。

这些点来自 `LND.json + handeye.yaml + robots.json`。

上一步重新生成后，第 0 帧结果是：

```text
fin-local LND keypoints: 左目/右目 21/24 可见
```

这说明：

```text
LND.json、handeye.yaml、robots.json 这条链是有用的。
它们不是废的。
PSM 的关键点能投影回真实图像附近。
```

所以我们没有放弃 grasp5 里的 LND 和 handeye。

### 4.2 URDF dense mesh 是什么

URDF dense mesh 是把完整 dVRK PSM 模型的网格表面密集采样成很多点。

上一步采样大约是：

```text
URDF dense mesh: 200390 个点
```

但是投影结果是：

```text
URDF dense mesh: 左目/右目 0/200390 可见
```

这说明完整 PSM 网格没有投到图像里。

### 4.3 为什么会出现这种矛盾

它不是矛盾，而是说明两套 PSM 使用的“基座坐标系”没有对齐。

可以这样理解：

```text
LND keypoints 用的是 LND/handeye 自己定义的 PSM base。
URDF mesh 用的是 dVRK xacro 展开出来的 PSM base/link 坐标。
```

如果这两个 base 的原点和方向不一样，那么会发生：

```text
LND 小点投影是对的。
但完整 URDF 网格整体被放错位置或转错方向。
```

这就是为什么左目/右目里 LND keypoints 能看到，但 URDF dense mesh 完全看不到。

换成更直白的话：

```text
我们已经知道“机械臂应该在图像哪里”。
但 dVRK xacro 里的完整机器人模型，还没有正确摆到那个位置。
```

---

## 5. 现在清掉了哪些数据

刚才删除了这些类型的数据：

```text
data/super/grasp5_native/depth/
data/super/grasp5_native/masks/
data/super/grasp5_native/projection_overlay_right_handed.png
data/super/grasp5_native/projection_right_handed.png
data/super/grasp5_native/scene_3d_right_handed.png
data/super/grasp5_native/psm_projection_right_handed_dense.png
data/super/grasp5_native/psm_scene_right_handed_dense.png
data/super/grasp5_offline_demo/bodies/
data/super/grasp5_offline_demo/tissue_debug/
data/super/grasp5_offline_demo/ground_debug/
examples/embodied_environments/super_embodied/objects/
examples/embodied_environments/super_embodied/environment/
data/super/psm_robot/psm_frame0_right_handed_dense.ply
```

这些内容要么明确记录了 super2 路径，要么是基于 super2 depth/mask 派生的组织、桌面和验证图，要么是旧 base 下生成的 PSM 点云/投影图。

删除它们后，当前 fin 目录不会再拿这些旧结果当作可信输入。

---

## 6. 哪些数据没有清掉

没有清掉原始数据：

```text
data/super/grasp5/
data/super/grasp5_native/rgb/
data/super/grasp5_native/robots.json
data/super/grasp5_native/timestamps*.json
data/super/grasp5_native/calib_rectified.json
```

没有清掉 fin 本地重建的 PSM/LND 数据：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_generation_report.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_projection_debug/
```

没有清掉 dVRK xacro 展开的完整 PSM 模型：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/meshes/*.stl
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm_rebuild_report.json
```

但要注意：

```text
psm.urdf 里的 fixed base 位姿目前还不能当最终正确值。
```

它需要下一步用 fin-local LND keypoints 对齐后重新确定。

---

## 7. 现在处在什么状态

当前状态可以总结为：

```text
原始图像、关节、标定还在。
PSM/LND 中间数据已经能在 fin 内生成。
LND keypoints 可以投影回图像。
完整 URDF mesh 还没有和 LND/handeye base 对齐。
组织/桌面/depth/mask 已经清掉，需要之后在 fin 内重新生成。
```

所以现在不是“所有都坏了”，而是进入了一个更干净的状态：

```text
先保留可信原始数据和 fin-local PSM/LND。
再从 fin 自己重新生成 depth/mask/tissue/ground。
最后解决完整 URDF PSM mesh 和 LND keypoints 的 base 对齐。
```

---

## 8. 下一步应该做什么

建议下一步按这个顺序来：

1. 重新在 fin 内生成 depth。
2. 重新在 fin 内生成 tissue/ground mask。
3. 用 fin 的 depth/mask 重建 tissue.json 和 ground.json。
4. 重新生成左右目 tissue/ground overlay，确认组织和桌面仍然对齐。
5. 用 fin-local `psm1_lnd_motion.json` 作为参考，对齐完整 dVRK URDF 的 base。
6. 重新生成 PSM dense 点云和左右目投影图。

最关键的是第 5 步：

```text
让 dVRK xacro URDF 的完整 mesh，跟 LND keypoints 指向的真实器械位置对上。
```

这一步做对后，才应该继续进入真正的 demo 组装和仿真。
