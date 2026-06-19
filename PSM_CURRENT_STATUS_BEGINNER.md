# PSM 当前情况说明（给初学者）

日期：2026-06-18

本文解释当前 SUPER / PSM 适配中看到的问题：为什么 `psm_assembled.ply` 里面有一些机器人零件，但新生成的 `psm_frame0_right_handed_dense.ply` 里面看不到这些零件。

## 2026-06-18 更新：当前状态

本文前半部分记录的是重建前的问题：旧 `psm.urdf` 里有些零件有 mesh，但没有 joint 接回机器人主链，所以 dense 点云漏掉部分零件。

这个问题现在已经处理：

```text
已从 dVRK xacro 完整重建 PSM。
新的 psm.urdf 有 18 个 link、17 个 joint。
所有 link 都可以从 world 遍历到。
两片夹爪的 visual 和 collision 都保留。
旧的 mimic 标签已导出到 psm_mimic_map.json。
```

新的关键文件：

```text
data/super/psm_robot/psm.urdf
data/super/psm_robot/psm_mimic_map.json
data/super/psm_robot/psm_rebuild_report.json
data/super/psm_robot/psm_frame0_right_handed_dense.ply
```

当前剩下的主要问题已经不是“点云缺零件”，而是：

```text
PSM 点云/关键点投影到左右目图像时仍然 mesh_visible=0。
这说明 PSM 和相机之间的坐标空间还没有对齐。
```

也就是说：

```text
URDF 结构问题：已修。
PSM 投影坐标问题：待继续修。
```

---

## 1. 现在我们已经完成了什么

当前工作目录是：

```text
/home/hsieh/data0/wad/embodied_gaussians_fixed_super_fin
```

目前已经完成的主要内容：

1. 已经把 SUPER 场景从原始相机坐标系转换到了右手系 table world。
2. 已经修正了之前的坐标系问题：不再使用会破坏旋转合法性的单轴镜像，而是使用合法的 180 度旋转方式。
3. 已经重新生成并检查过组织和左右目图像的投影验证图。
4. 目前组织在左目和右目看起来都已经比较吻合。
5. 现在进入 PSM 机器人专项验证阶段。

目前关键输出包括：

```text
data/super/grasp5_native/scene_3d_right_handed.png
data/super/grasp5_native/projection_right_handed.png
data/super/grasp5_native/projection_overlay_right_handed.png
data/super/psm_robot/psm_assembled.ply
data/super/psm_robot/psm_frame0_right_handed_dense.ply
data/super/grasp5_native/psm_projection_right_handed_dense.png
data/super/grasp5_native/psm_scene_right_handed_dense.png
```

其中：

- `psm_assembled.ply`：早期生成的 PSM 静态拼装点云。
- `psm_frame0_right_handed_dense.ply`：现在根据 URDF 和第 0 帧关节角重新生成的 PSM 点云。

你现在看到的问题是：

```text
psm_assembled.ply 里有的关节/零件，
psm_frame0_right_handed_dense.ply 里没有。
```

这个现象是合理的，原因不是“采样点不够密”，而是 URDF 里面有一些零件没有真正连接到机器人结构树上。

## 2. 先解释几个基本概念

### 2.1 什么是 mesh

`mesh` 可以理解成一个零件的三维模型。

比如：

```text
meshes/outer_yaw.stl
meshes/tool_main.stl
meshes/tool_wrist_sca_link_2.stl
```

这些 `.stl` 文件只是“零件的形状”。

它们本身不知道：

- 自己应该放在哪里；
- 应该跟着哪个关节运动；
- 应该连接到机器人哪一段；
- 当前帧应该是什么姿态。

所以 mesh 只是形状文件，不等于完整机器人。

### 2.2 什么是 link

`link` 可以理解成“机器人上的一个刚体部件”。

在 URDF 里，一个 link 通常会包含一个 mesh：

```xml
<link name="PSM1_tool_wrist_sca_ee_link_1">
  <visual>
    <geometry>
      <mesh filename="meshes/tool_wrist_sca_link_2.stl"/>
    </geometry>
  </visual>
</link>
```

这表示：

```text
有一个 link，名字叫 PSM1_tool_wrist_sca_ee_link_1；
它显示出来时使用 tool_wrist_sca_link_2.stl 这个三维模型。
```

但是，仅仅写了 `<link>` 还不够。

它还必须通过 `<joint>` 接到机器人主结构上，否则它只是一个孤立零件。

### 2.3 什么是 joint

`joint` 可以理解成“两个 link 之间的连接方式”。

例如：

```xml
<joint name="yaw" type="revolute">
  <parent link="PSM1_psm_base_link"/>
  <child link="PSM1_outer_yaw_link"/>
</joint>
```

意思是：

```text
PSM1_outer_yaw_link 连接在 PSM1_psm_base_link 上；
连接方式是 revolute，也就是旋转关节；
这个关节名字叫 yaw。
```

机器人之所以能从底座一路算到末端，就是因为 link 之间有 joint 串起来。

### 2.4 什么是 fixed joint

`fixed joint` 是“不动的连接”。

例如：

```xml
<joint name="some_part_fixed" type="fixed">
  <parent link="A"/>
  <child link="B"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

意思是：

```text
B 固定在 A 上；
B 不会相对 A 独立运动；
A 怎么动，B 就跟着怎么动。
```

fixed joint 不会增加新的活动关节，也不会改变 `robots.json` 里面的 7 个关节角。

### 2.5 什么是 FK

FK 是 forward kinematics，中文一般叫“正向运动学”。

不用怕这个词，它的意思很简单：

```text
已知每个关节当前的角度/位移，
从机器人底座开始，
一节一节往下算，
算出每个零件在世界坐标系里的位置和朝向。
```

例如：

```text
world
  -> base
    -> yaw
      -> pitch
        -> insertion
          -> wrist
            -> jaw
```

如果某个 link 没有 joint 接到这棵树上，FK 就走不到它。

FK 走不到它，当前 dense 脚本就不会采样它。

## 3. 为什么 psm_assembled 有，但 dense 没有

这两个文件生成方式不同。

### 3.1 psm_assembled.ply 是静态拼装

`psm_assembled.ply` 更像是：

```text
把所有 mesh 文件按照一个预设方式拼在一起，
直接采样成点云。
```

它不严格依赖 URDF 的 joint 树。

所以即使某些零件在 URDF 里面没有接到主链上，静态拼装时仍然可能把它们放进去。

因此 `psm_assembled.ply` 里面能看到更多零件。

### 3.2 psm_frame0_right_handed_dense.ply 是按 URDF + 第 0 帧关节生成

`psm_frame0_right_handed_dense.ply` 是现在更严格的验证产物。

它的生成逻辑是：

1. 读取 `data/super/psm_robot/psm.urdf`。
2. 从 `world` 开始，根据 joint 关系遍历机器人结构。
3. 读取 `robots.json` 第 0 帧的 7 个关节值。
4. 对每个能被 FK 遍历到的 link，找到它的 visual mesh。
5. 对这些 mesh 做密集采样。
6. 输出新的点云。

所以它只采样“真正接在 URDF 主树上的零件”。

如果一个 link 有 mesh，但是没有 joint 连接到主树，那么 dense 脚本会跳过它。

这就是现在缺零件的直接原因。

## 4. 当前 URDF 里面缺了哪些连接

当前 `psm.urdf` 里面有一些 link 是“有 mesh，但是没有父 joint”。

这些 link 包括：

```text
PSM1_tool_wrist_sca_ee_link_1
PSM1_tool_wrist_sca_ee_link_2
PSM1_outer_insertion_link
PSM1_outer_pitch_back_link
PSM1_outer_pitch_front_link
PSM1_outer_pitch_bottom_link
PSM1_outer_pitch_top_link
```

它们的问题不是没有 mesh。

它们的问题是：

```text
URDF 里没有 joint 把它们挂到主机器人链条上。
```

所以现在情况可以理解成：

```text
零件模型存在；
零件名字也存在；
但是没有告诉机器人：这个零件应该接在哪里。
```

## 5. 为什么之前注释写着“保留为 fixed 子 link”，但实际还是缺

URDF 文件开头的注释大意是：

```text
有些并行机构和 mimic 关节不作为活动关节处理；
它们应该作为 fixed 子 link 保留下来用于渲染。
```

这个想法是对的。

但是实际文件里只保留了 link，没有补上对应的 fixed joint。

也就是说，文件现在相当于：

```text
计划：这些装饰零件要用 fixed joint 挂回主链。
现实：link 和 mesh 在，但 fixed joint 没写进去。
```

所以 dense 点云里会缺这些零件。

## 6. 如果补 actual joint，会发生什么

这里的 “actual joint” 我建议先理解为：

```text
在 URDF 里真的写入 joint，让这些 link 不再孤立。
```

第一阶段最稳的是补 fixed joint。

例如：

```xml
<joint name="outer_pitch_back_fixed" type="fixed">
  <parent link="PSM1_outer_pitch_link"/>
  <child link="PSM1_outer_pitch_back_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

这样 `PSM1_outer_pitch_back_link` 就会跟着 `PSM1_outer_pitch_link` 运动。

这不会新增活动自由度。

也就是说：

```text
robots.json 里还是 7 个关节；
仿真里还是 7 个主要活动关节；
只是视觉零件更完整了。
```

## 7. 我建议怎么补

建议先补以下 fixed joint：

```text
PSM1_outer_pitch_link
  -> PSM1_outer_insertion_link
  -> PSM1_outer_pitch_back_link
  -> PSM1_outer_pitch_front_link
  -> PSM1_outer_pitch_bottom_link
  -> PSM1_outer_pitch_top_link
```

这些是外部并行机构/装饰结构。

还要补夹爪视觉零件：

```text
PSM1_tool_wrist_sca_ee_link_0
  -> PSM1_tool_wrist_sca_ee_link_1
  -> PSM1_tool_wrist_sca_ee_link_2
```

或者也可以把夹爪两个视觉 link 挂到：

```text
PSM1_tool_wrist_sca_shaft_link
```

这两个选择的区别是：

- 挂到 `PSM1_tool_wrist_sca_ee_link_0`：会跟随 jaw 关节的旋转锚点。
- 挂到 `PSM1_tool_wrist_sca_shaft_link`：不会跟随 jaw 开合，只跟随腕部末端。

目前更符合“保留 jaw 视觉零件”的做法是先挂到 `PSM1_tool_wrist_sca_ee_link_0`。

但是注意：现在这样仍然不是完整 mimic jaw，只是让夹爪零件不再消失。

## 8. 主要难点在哪里

### 难点 1：不能破坏 7 个关节的 q 映射

当前 `robots.json` 里面的 PSM 关节值是 7 个：

```text
yaw
pitch
insertion
roll
wrist_pitch
wrist_yaw
jaw
```

我们不能随便新增活动关节，否则会出现：

```text
URDF 需要 9 个或更多关节值，
但 robots.json 只有 7 个。
```

所以第一阶段只补 fixed joint，是最安全的。

### 难点 2：夹爪原本可能是 mimic 结构

真实 PSM 夹爪不是只有一个简单零件。

它通常有两个夹爪片，而且两个夹爪片会按相反方向开合。

这种关系在 URDF 里经常用 mimic joint 表达。

但是 Warp 这类仿真加载器未必支持 mimic joint。

所以之前适配时把 mimic 关节删掉了，只保留 7 个主要活动关节。

如果我们现在要恢复“真实夹爪开合”，就不是简单补 fixed joint 了，而是要：

1. 给两个夹爪片各自建立关节。
2. 手动规定它们如何跟随 `jaw`。
3. 修改 FK/仿真加载逻辑，让一个 `jaw` 值驱动多个视觉子关节。
4. 检查 Warp 是否接受这种额外结构。

这会影响比较大。

所以建议先不要做这一步。

### 难点 3：joint origin 不能乱加

每个 link 的 visual 里已经有自己的 `<origin>`。

例如：

```xml
<visual>
  <origin rpy="0 -1.5708 1.5708" xyz="0.02528 0.429 0"/>
  <geometry>
    <mesh filename="meshes/outer_insertion.stl"/>
  </geometry>
</visual>
```

这说明 mesh 相对 link 已经有一段偏移和旋转。

如果我们补 fixed joint 时又加一段偏移，就可能把这个零件移动两次。

所以第一版补 joint 时，建议使用：

```xml
<origin xyz="0 0 0" rpy="0 0 0"/>
```

这样只建立连接，不额外移动零件。

然后通过重新生成点云来检查是否和 `psm_assembled.ply` 一致。

### 难点 4：PSM 投影本身还有坐标空间问题

这和“缺零件”是两个不同问题。

缺零件问题来自 URDF 拓扑：

```text
有些 link 没有 joint 接回主树。
```

PSM 投影问题来自相机/PSM 坐标空间：

```text
当前 PSM 点云投影到左右目图像时，还没有很好地落到图像中的工具位置。
```

之前检查发现：

- tissue/ground 的右手系转换已经基本正确；
- 左右目组织投影已经比较吻合；
- 但是 PSM 的 LND / handeye / rectified camera 之间可能还存在坐标空间不一致；
- 有些 PSM 数据在 `raw_camera_tuned` 空间里更像是可见的，但在 `rectified_camera` 空间里不可见。

所以后面需要分开处理：

1. 先修 URDF 拓扑，让 dense 点云零件完整。
2. 再继续修 PSM 到相机图像的投影对齐。

不要把这两个问题混在一起。

## 9. 建议下一步

我建议下一步按这个顺序做：

1. 备份当前 `psm.urdf`。
2. 只在 `embodied_gaussians_fixed_super_fin/data/super/psm_robot/psm.urdf` 里补 fixed joint。
3. 不修改 `robots.json` 的 7 个关节顺序。
4. 重跑 `scripts/generate_super_psm_validation.py --points 200000`。
5. 检查新生成的 `psm_frame0_right_handed_dense.ply` 是否包含之前缺失的零件。
6. 把每个 link 的采样点数量打印出来，确认不再漏 link。
7. 如果点云形状完整，再继续处理 PSM 投影到左右目图像不对齐的问题。

## 10. 当前结论

当前 `psm_frame0_right_handed_dense.ply` 缺少部分零件，不是因为采样太稀，也不是因为 mesh 文件丢了。

真正原因是：

```text
URDF 里面有几个有 mesh 的 link 没有 joint 接到机器人主链。
```

因此 dense 脚本按 FK 遍历时走不到这些 link，就没有采样它们。

最稳的修复方式是：

```text
给这些孤立 link 补 fixed joint，
让它们成为机器人结构树的一部分，
但不新增活动关节，
不改变 7 个 q 的映射。
```

这样可以先解决“点云缺零件”的问题。

之后再继续解决更大的 PSM 投影坐标对齐问题。
