# PSM URDF Base 对齐说明（给初学者）

日期：2026-06-19

本文解释三个问题：

```text
1. 现在 URDF base 是从哪里来的？
2. 为什么它现在不一定可信？
3. 后面怎么调整 URDF base，让完整器械模型贴近 LND keypoints 和真实图像？
```

---

## 1. 先用一句话说清楚

完整 dVRK URDF 像一套“精细机器人模型”，LND 像一套“能在真实相机图像里找到器械位置的简化骨架”。

现在的问题是：

```text
精细机器人模型本身是完整的，
但这套模型摆在世界里的起点位置，也就是 URDF base，还没和真实图像对齐。
```

所以我们下一步不是重新做 tissue/ground，也不是重新分割 mask，而是要做：

```text
调整 URDF base
让完整 URDF mesh 投影到相机图像后，贴近 LND keypoints 和真实器械。
```

---

## 2. 什么是 URDF base

URDF 是机器人模型文件。

里面有很多 link 和 joint，例如：

```text
base link
yaw link
pitch link
insertion link
wrist link
jaw links
```

这些 link 串起来就是一条 PSM 机械臂。

但是模型里面的坐标只是“机器人自己身体内部的坐标”。它还需要一个起点告诉系统：

```text
这台机器人在我们的世界坐标里，放在哪里？朝哪个方向？
```

这个起点就是 URDF 里的 fixed joint：

```xml
<joint name="fixed" type="fixed">
  <parent link="world"/>
  <child link="PSM1_psm_base_link"/>
  <origin rpy="..." xyz="..."/>
</joint>
```

可以把它理解成：

```text
world 坐标系里的一个固定安装座。
PSM 整台机器人都挂在这个安装座上。
```

如果这个安装座的位置或方向错了，后面所有 link、夹爪、mesh 都会一起错。

---

## 3. 现在 URDF base 是从哪里来的

当前 `data/super/psm_robot/psm.urdf` 里有一个 fixed joint，大概是：

```xml
<origin rpy="-0.238342 -0.478597 2.466444"
        xyz="0.132451 -0.155286 0.139112"/>
```

这个值不是 dVRK xacro 天然给出的真实安装位置。

它来自之前旧阶段的推导：

```text
旧 PSM/LND 中间数据
+ 旧 table/rectified 坐标变换
→ 推出一个 world -> PSM base 的位姿
```

当时这样做是为了先让 PSM 大概进入场景。

但现在我们已经重新做了很多关键东西：

```text
人工 SAM2 mask
新的 depth
新的 tissue/ground 点云
新的 ground plane
新的 right-handed table-world
新的 cameras.json
```

也就是说，“世界坐标系”本身已经重新整理过。

所以旧的 URDF base 很可能已经不再适合当前世界坐标。

---

## 4. 为什么 URDF base 错了会很严重

可以用一个很直观的比喻。

假设你有一个很精细的手术器械模型，它的每个零件都对：

```text
夹爪形状对
腕部形状对
杆子长度对
关节连接对
```

但是你把整台机器人放错了地方，比如：

```text
向左偏了 10cm
向上偏了 5cm
转了 90 度
甚至放到了相机背后
```

那么再精细的模型也会投影错。

这就是当前 PSM 的核心问题：

```text
URDF 结构基本完整，
但 URDF base 还没和真实图像里的 PSM 对齐。
```

---

## 5. LND 在这里起什么作用

LND 可以理解成“简化版 PSM 骨架”。

它不是最终要用于物理仿真的完整机器人，但它有一个重要优点：

```text
LND + handeye + robots.json 算出来的 keypoints，
能比较合理地投影回真实相机图像。
```

所以 LND 现在像一个参考标尺。

我们可以认为：

```text
LND keypoints 告诉我们：真实器械大概应该在图像哪里。
URDF mesh 告诉我们：完整器械模型长什么样。
```

下一步要做的就是：

```text
把完整 URDF mesh 的 base 调整到 LND keypoints 指示的位置。
```

---

## 6. 为什么不能直接相信 dVRK xacro

dVRK xacro 给的是机器人结构：

```text
link 怎么连
joint 怎么转
mesh 文件在哪里
每个零件相对上一个零件在哪里
```

但它通常不知道我们这次实验里：

```text
相机在哪里
桌面在哪里
PSM 实际安装在哪里
grasp5 数据集里的 handeye 是怎么标定的
```

所以 dVRK xacro 能告诉我们“机器人身体内部怎么长”，但不能直接告诉我们“这台机器人在当前视频里的世界坐标应该放哪里”。

这个“放哪里”的问题，就是 URDF base alignment。

---

## 7. 接下来怎么调整 URDF base

整体思路是：

```text
用 LND keypoints 作为真实参考，
反过来求完整 URDF 应该放在哪里。
```

具体步骤如下。

---

## 8. 第一步：重新确认当前 table-world 下的 LND

因为我们刚刚重新生成了 ground plane 和 table-world，所以 PSM/LND 的 table 坐标也要跟当前世界一致。

要重新生成或检查：

```text
data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json
data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json
```

这些文件里应该保存：

```text
第 0 帧 LND keypoints 在当前 table-world 下的位置
```

这些点是后面校准 URDF base 的目标参考。

---

## 9. 第二步：先让 URDF 在自己坐标里动起来

从 `robots.json` 读取第 0 帧关节值。

然后用这些关节值驱动 URDF。

因为原始数据只有 7 个活动关节，完整 URDF 里还有 mimic joint，所以要用：

```text
data/super/psm_robot/psm_mimic_map.json
```

把 7 个主关节展开成完整 URDF 需要的关节。

例如：

```text
jaw -> jaw_mimic_1
jaw -> jaw_mimic_2
pitch -> pitch mimic links
```

这一步得到的是：

```text
URDF mesh / link / jaw 在 URDF base 坐标系下的位置
```

注意：此时还没有真正放进 table-world。

---

## 10. 第三步：找到 URDF 和 LND 的对应点

要做对齐，需要两边有一些可以对应的点。

例如：

```text
LND 的 wrist keypoint   ↔ URDF 的 wrist link 附近点
LND 的 jaw keypoint     ↔ URDF 的 jaw link 附近点
LND 的 tool tip keypoint ↔ URDF 的 tool tip / jaw tip 附近点
```

如果对应点足够清楚，就可以用数学方法直接求一个刚体变换。

这个刚体变换就是：

```text
T_table_urdf_base
```

它表示：

```text
URDF base 应该放在 table-world 的哪里，朝哪个方向。
```

---

## 11. 第四步：用刚体配准求 base correction

数学上可以这样写：

```text
T_table_urdf_base @ URDF_local_points ≈ LND_table_points
```

意思是：

```text
把 URDF 自己坐标里的点，
用一个 base 变换搬到 table-world，
让它们尽量接近 LND 的点。
```

如果有明确的一一对应点，可以用 Kabsch/SVD 求最优旋转和平移。

通俗说就是：

```text
找一个最合理的平移 + 旋转，
让两组点尽量重合。
```

---

## 12. 第五步：写回 psm.urdf

求出新的 `T_table_urdf_base` 后，需要把它转换成 URDF fixed joint 里的：

```text
xyz
rpy
```

然后写回：

```text
data/super/psm_robot/psm.urdf
```

替换当前旧的 fixed joint。

也就是更新：

```xml
<joint name="fixed" type="fixed">
  <parent link="world"/>
  <child link="PSM1_psm_base_link"/>
  <origin rpy="新rpy" xyz="新xyz"/>
</joint>
```

---

## 13. 第六步：生成验证图

这一步非常关键，不能只看数字。

需要生成：

```text
左目图像 + URDF mesh overlay
右目图像 + URDF mesh overlay
左目图像 + LND keypoints + URDF mesh overlay
table-world 3D 图：camera / tissue / ground / PSM
PSM dense point cloud PLY
```

判断标准：

```text
URDF mesh 不在相机后面
左右目都能看到器械
夹爪和腕部大致贴近真实图像
URDF mesh 和 LND keypoints 大体重合
PSM 和 tissue/ground 的相对位置合理
```

如果这些验证不通过，就不能进入 demo。

---

## 14. 为什么这一步不需要 mask

PSM base 对齐主要依赖几何关系：

```text
robots.json
LND.json
handeye.yaml
psm.urdf
psm_mimic_map.json
cameras.json
calib_rectified.json
```

mask 不是必须的。

mask 只有在我们想自动评估器械轮廓时才有用，例如：

```text
URDF 投影轮廓 和 手工器械 mask 做 IoU
```

但当前阶段先不需要。

现在最重要的是让 URDF mesh 先出现在正确位置。

---

## 15. 当前风险点

### 风险 1：LND 和 URDF 的点不一定一一对应

LND 是简化骨架，URDF 是完整模型。

它们的 keypoint/link 名字和位置可能不是完全一致。

所以第一版可能只能做粗对齐。

### 风险 2：jaw / mimic 需要正确展开

如果夹爪 mimic 展开错了，夹爪会张反、位置错、甚至两片夹爪重合。

所以必须检查：

```text
jaw_mimic_1
jaw_mimic_2
```

是否从 `jaw` 正确派生。

### 风险 3：当前 table-world 刚重建过

我们刚刚重新做了 tissue/ground/table-world。

所以所有旧的 PSM base 结论都要谨慎，不能直接复用。

---

## 16. 最后总结

当前 URDF base 是旧推导值，不是最终可信值。

LND 的作用是提供一个更贴近真实图像的 PSM 关键点参考。

下一步要做的是：

```text
用当前 table-world 下的 LND keypoints
校准完整 dVRK URDF 的 base
让 URDF mesh 投影到图像里时贴近真实器械
```

这一步完成后，PSM 才能真正接入后面的 demo 和仿真。
