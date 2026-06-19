# 下一步计划：PSM 机械臂接入

> 写于 2026-06-17。截至此文档，Phase 1-8 已完成，Phase 9-10 待做。

---

## 一、我们有什么（已完成的工作）

### 数据和场景

| 文件 | 内容 | 坐标系 |
|------|------|--------|
| `robots.json` | 5458 帧关节角，每帧 7 个值 | — |
| `cameras.json` | 左右双相机（X_WC） | 桌面坐标系 |
| `stereo_left/right.mp4` | 1644 帧视频 | 图像 |
| `tissue.json` | 组织，1991 个 particle + 1991 个 gaussian | 桌面坐标系，z≥0，X_WB=I |
| `ground.json` | 桌面，2000 个 particle + 2000 个 gaussian | 桌面坐标系，z=0，X_WB=I |
| `ground_plane.json` | 物理碰撞面 `[0,0,1,0]`，即 z=0 水平面 | 桌面坐标系 |

### PSM 机械臂

| 文件 | 内容 |
|------|------|
| `psm.urdf` | 7 个活动关节（6 revolute + 1 prismatic），基座位姿已写入 fixed 关节 |
| `meshes/*.stl` | 14 个 STL 部件（从 DAE 转换） |
| `psm_mimic_map.json` | mimic 关节映射表（jaw 左右夹爪的反向运动） |
| `psm_rebuild_report.json` | URDF 重建报告 |

### 坐标系

世界坐标系 = 桌面坐标系（table_world_z_aligned_m），桌面在 z=0，Z 轴朝上，重力 (0,0,-9.8)。

---

## 二、当前问题

### 问题 1：PSM 投影和图像不重合

我们用 super2 的 LND 关键点（器械上的 24 个特征点）投影到左相机 RGB 图像上，**12/24 个点在图像内，但它们的位置和图像中实际的器械不重合**。

**原因**：`psm.urdf` 的基座位姿是从 super2 的 `T_rectified_camera_psm_base` 推算的，但这个位姿可能和实际器械的真实位置有偏差。dVRK xacro 的基座定义和 grasp5 的 LND/handeye 基座定义**不是同一个坐标系**。

```
dVRK xacro 说的 "基座" ≠ LND/handeye 说的 "基座"
```

所以直接把 URDF 放进 table-world 不够——需要做**基座修正（base correction）**。

### 问题 2：ceramic 尖端关键点只是点，没有完整 mesh

LND 关键点只给出 24 个离散点，不是完整的器械 mesh。完整 mesh 在 URDF 的 STL 文件里。我们需要**把 URDF mesh 移动到和 LND 关键点一致的位置**。

---

## 三、解决思路

### 核心方法：计算 base correction transform

```
第 0 帧:
  1. 用 LND.json + handeye.yaml + robots.json → 计算器械关键点在桌面坐标系中的位置
  2. 用 psm.urdf + robots.json → 计算 URDF 器械末端在桌面坐标系中的位置
  3. 比较两者差距 → 算出一个修正变换 T_correction
  4. 把 T_correction 乘到 URDF 的 fixed joint 上
```

**类比**：URDF 是一个机器人模型，LND 告诉你「真正的机器人手指尖在这里」。你发现模型的手指和真实手指差了 5cm。于是你把整个模型平移+旋转 5cm，让模型手指对准真实手指。

### 不需要的东西

- **不需要 tissue/ground mask**：PSM 对齐只涉及 URDF + LND + 关节角 + 相机
- **不需要深度图**：只需要投影到 2D 验证
- **不需要 SAM2**：这是纯几何对齐问题

---

## 四、具体步骤

### Step 1：写诊断脚本

`scripts/diagnose_psm_alignment.py`

这个脚本做：

1. 读取 robots.json 第 0 帧关节角
2. 用 **LND.json + handeye.yaml** 算器械关键点的桌面坐标（已有 super2 数据可参考）
3. 用 **psm.urdf + 第 0 帧关节角** 算 URDF 末端 link 的桌面坐标（FK）
4. 对比两者差距，算出初步的 `T_correction`
5. 把修正后的 URDF mesh 和 LND 关键点一起投影到左右相机图像上
6. 输出对比图

### 输出验证

```
data/super/grasp5_native/
├── psm_alignment_left.png    — 左相机：URDF(红) + LND关键点(绿) + RGB背景
├── psm_alignment_right.png   — 右相机：同上
└── psm_alignment_report.json — 数值报告（距离差、修正变换矩阵）
```

要求：
- 左右相机都能看到器械
- 夹爪位置贴合图像
- PSM 不跑到相机后面

### Step 2：写入 URDF

对齐验证通过后，把 `T_correction` 乘到 `psm.urdf` 的 fixed joint 上，更新 `rpy` 和 `xyz`。

### Step 3：接入 Demo（Phase 9）

写 `super_embodied.py`，包含：
- PSM URDF（对齐后的基座位姿）
- tissue（rigid body，受 visual forces）
- ground（visual body only）
- ground_plane（物理碰撞）
- 仅尖端 link 放高斯（wrist_sca, jaw links）

### Step 4：运行测试（Phase 10）

写 `example_embodied_super_offline.py`，跑起来修 bug。

---

## 五、和之前计划的对比

| | 旧计划 | 新计划 |
|------|--------|--------|
| Phase 8 PSM 基座 | 直接从 super2 拿坐标 | **先诊断对齐，再修正** |
| Phase 9 super_embodied | 直接写 | **对齐后再写** |
| 是否需要 mask | 不需要 | 不需要 |
| 验证方式 | 单相机投影 | **双相机投影 + 定量报告** |

核心区别：旧计划直接把 super2 的坐标写死，新计划先算修正再写入。

---

## 六、难点

1. **坐标系不一致**：dVRK xacro 和 LND/handeye 的基座定义不同。不能假设它们指向同一个点。
2. **prismatic 关节**：insertion 是平移关节（单位米），和 6 个旋转关节（单位弧度）混在一起。FK 计算需要正确处理。
3. **mimic 关节**：URDF 中有 jaw_mimic_1 和 jaw_mimic_2，它们跟随 jaw 运动但方向相反（模拟夹爪开合）。robots.json 只有 jaw 一个关节值，mimic 需要由 psm_mimic_map.json 派生。
4. **验证成本**：需要在左右相机两张图上都能看到器械，且夹爪位置合理。
