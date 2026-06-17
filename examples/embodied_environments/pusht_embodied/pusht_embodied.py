# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json
from pathlib import Path

import numpy as np
import warp
import warp.sim
import torch


from embodied_gaussians import (Body, Ground, VirtualCamerasBuilder, EmbodiedGaussiansBuilder, EmbodiedGaussiansEnvironment, read_extrinsics, read_ground)

# 机械臂初始关节角。
# 如果你后面换成别的数据集，而那个数据集对应的是另一套机器人初始姿态，
# 这里往往也需要一起调整，否则一启动时机器人姿态就会和离线数据不一致。
Q_START = np.array(
    [
        2.6537935,
        1.5346614,
        -1.6750929,
        -2.61287848,
        1.46183864,
        1.54922744,
        0.82306163,
    ]
)
current_dir = Path(__file__).parent
# 这里定义了场景资源文件的位置：
# - ground_plane.json: 地面平面参数
# - extrinsics.json: 相机外参
# - objects/*.json: 物体的粒子/高斯表示
GROUND_PATH = current_dir / Path("environment/ground_plane.json")
EXTRINSICS_PATH = current_dir / Path("environment/extrinsics.json")
BODY_NAME = "tblock"
BODY_ID = warp.constant(13)


def get_body(name: str) -> Body:
    # 读取一个物体的 embodied-gaussian 描述文件。
    # 这些 json 一般定义了：
    # 1. 物体的粒子/刚体结构
    # 2. 高斯渲染所需的外观参数
    #
    # 如果你换别的数据集，且场景里物体不是 tblock / ground，
    # 最先需要替换的通常就是这里读取的对象描述文件。
    path = current_dir / Path(f"objects/{name}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return Body.model_validate(data)


def build_environment(num_envs: int = 1, add_gaussians: bool = True):
    # 这个函数负责“搭建仿真世界”。
    # 它和离线数据集不是一回事：
    # - 这里定义的是世界里有哪些对象、机器人、地面、渲染高斯
    # - 数据集则决定这些对象在每个时间戳应该处于什么状态、看到什么图像
    #
    # 所以如果你想让“别的数据集”跑起来，必须先确认：
    # 1. 这个环境里的机器人/物体，是否和数据集对应
    # 2. 数据集记录下来的关节数、相机数、相机位姿，是否能和这里对上
    ground_data = read_ground(GROUND_PATH)
    ground = Ground(plane=ground_data)
    body = get_body(BODY_NAME)
    ground_body = get_body("ground")
    builder = EmbodiedGaussiansBuilder()
    if add_gaussians:
        # add_renderable_articulation_from_urdf 会同时加入：
        # - 物理仿真所需的机器人结构
        # - 可渲染的 Gaussian / visual 表示
        builder.add_renderable_articulation_from_urdf(
            urdf_path=current_dir / Path("assets/robots/panda.urdf"),
            initial_joints=Q_START,
            stiffness=500,
            damping=100,
            ignore_inertial_definitions=True,
            ensure_nonstatic_links=True,
            collapse_fixed_joints=False,
        )
    else:
        # 这个分支只加物理结构，不加高斯渲染表示。
        # 一般在只关心动力学、不关心图像渲染时才会用到。
        builder.add_articulation_from_urdf(
            urdf_path=current_dir / Path("assets/robots/panda.urdf"),
            initial_joints=Q_START,
            stiffness=500,
            damping=100,
            ignore_inertial_definitions=True,
            ensure_nonstatic_links=True,
            collapse_fixed_joints=False,
        )

    # 加入主要操作物体（默认是 T-block）。
    bid = builder.add_rigid_body(body, add_gaussians=add_gaussians)
    if add_gaussians:
        # 地面也作为可视化对象加入，这样渲染时能看到它。
        builder.add_visual_body(ground_body)
    final_builder = EmbodiedGaussiansBuilder()
    final_builder.set_ground_plane(ground.normal(), ground.offset())
    for env in range(num_envs):
        # 多环境模式下，会把同一套场景复制多份。
        final_builder.add_builder(builder)
    env = EmbodiedGaussiansEnvironment(final_builder)
    # 这里把前 13 个 body 的 gravity_factor 设成 0，
    # 相当于让某些主体不受重力影响。
    # 如果你换场景后发现物体/机械臂行为异常，这里也值得检查。
    warp.to_torch(env.sim.model.gravity_factor).reshape(num_envs, -1)[:, :13] = 0.0
    q_start = torch.from_numpy(Q_START).float()
    # 初始化机器人控制目标，使其和 Q_START 一致。
    env.sim.get_joint_act()[:, :7] = q_start
    env.stash_state()
    return env


def add_static_cameras(builder: VirtualCamerasBuilder):
    # 这里定义“静态相机”的内参 K。
    # 注意：这是仿真里的虚拟相机，不是离线数据集目录里的 cameras.json。
    #
    # 如果你要让别的数据集和这里对齐，通常需要保证：
    # 1. 数据集里的相机内参与这里一致，或者
    # 2. 你把这里改成和数据集相机一致
    K = np.array(
        [
            [241.0, 0.0, 240.0],
            [0.0, 241.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # 每个条目是一个相机：
    # - 第一个值是相机 ID
    # - 第二个值是外参矩阵 X_BV
    #
    # 这些值如果和你离线数据集记录下来的相机位姿不一致，
    # 后面用 visual forces 做图像对齐时通常会效果很差，甚至完全错位。
    camera_data = [
        # (
        #     "220422302296",
        #     np.array(
        #         [
        #             [
        #                 0.606747567653656,
        #                 -0.5086942911148071,
        #                 0.6108089089393616,
        #                 0.9495939016342163,
        #             ],
        #             [
        #                 0.779473066329956,
        #                 0.5313830971717834,
        #                 -0.33174341917037964,
        #                 0.013426600024104118,
        #             ],
        #             [
        #                 -0.15581756830215454,
        #                 0.6773936152458191,
        #                 0.7189289331436157,
        #                 0.38001641631126404,
        #             ],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     ),
        # ),
        (
            "234222302164",
            np.array(
                [
                    [
                        -0.5385846495628357,
                        -0.7674259543418884,
                        0.347827672958374,
                        0.6469128727912903,
                    ],
                    [
                        0.8420496582984924,
                        -0.504772961139679,
                        0.19014911353588104,
                        0.5172103047370911,
                    ],
                    [
                        0.02964870259165764,
                        0.39529949426651,
                        0.9180736541748047,
                        0.5189733505249023,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # (
        #     "234222303707",
        #     np.array(
        #         [
        #             [
        #                 -0.0959780216217041,
        #                 0.971773624420166,
        #                 -0.21550923585891724,
        #                 0.17146411538124084,
        #             ],
        #             [
        #                 -0.9483034610748291,
        #                 -0.023470992222428322,
        #                 0.3164959251880646,
        #                 0.5118779540061951,
        #             ],
        #             [
        #                 0.30250418186187744,
        #                 0.23474480211734772,
        #                 0.9237889647483826,
        #                 0.7177857756614685,
        #             ],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     ),
        # ),
    ]
    for camera_id, X_BV in camera_data:
        builder.add_camera(camera_id, K, X_BV)


def add_gripper_camera(builder: VirtualCamerasBuilder):
    # 在夹爪上再绑一个相机。
    # body_id=12 表示这台相机会跟着指定的 robot link 一起动。
    #
    # 如果你的新数据集没有“手眼相机”，可以不加；
    # 如果有，就要确认 body_id 和安装位姿 X_BV 是否匹配。
    K = np.array(
        [
            [100.0, 0.0, 240.0],
            [0.0, 100.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
    )
    X_BV = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    builder.add_camera("gripper_camera", K, X_BV, body_id=12)
