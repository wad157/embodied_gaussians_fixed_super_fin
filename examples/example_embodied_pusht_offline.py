# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse
import os
import sys
import trio
from pathlib import Path

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent

# 强制优先导入当前仓库里的源码，而不是系统里已经安装的旧版本包。
# 否则即使在 embodied_gaussians_fixed 里改了代码，运行时也可能仍然走 embodied_gaussians4/src。
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(current_dir))

import warp as wp
import torch
from marsoom import imgui
from embodied_environments.pusht_embodied.pusht_embodied import build_environment
from embodied_gaussians import DatasetManager, EmbodiedGaussiansEnvironment
from embodied_gaussians.vis import EmbodiedGUI  


def parse_args():
    # 通过命令行参数选择要读取的离线数据集目录。
    # 这样以后切换数据集时，不需要再手改源码。
    parser = argparse.ArgumentParser(
        description="运行 PushT 离线回放示例，并按需切换数据集目录。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "离线数据集目录。目录内需要包含 robots.json、cameras.json、videos/*.mp4、videos/*.json。"
            "如果不传，就优先读取环境变量 EMBODIED_GAUSSIANS_DATASET，"
            "都没有时退回到仓库自带的 sample_demos/0。"
        ),
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: Path | None) -> Path:
    default_dataset = current_dir / Path("embodied_environments/pusht_embodied/sample_demos/0")

    # 命令行参数优先级最高；如果没传，就尝试读环境变量；
    # 再没有，就使用仓库里原本自带的示例数据集。
    dataset_from_env = os.environ.get("EMBODIED_GAUSSIANS_DATASET")
    dataset_path = dataset_arg
    if dataset_path is None and dataset_from_env:
        dataset_path = Path(dataset_from_env).expanduser()
    if dataset_path is None:
        dataset_path = default_dataset

    # 允许传相对路径，统一转成绝对路径，避免从不同工作目录启动时读错地方。
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    return dataset_path


class PlaybackControls:
    def __init__(self, environment: EmbodiedGaussiansEnvironment, dataset_manager: DatasetManager, fps: int):
        # 这个控制器维护一个独立的“数据集时间”游标。
        # 仿真器里的物理时间会持续向前推进，但机器人目标关节和相机观测帧
        # 是按照 current_timestep 从离线数据集中采样出来的。
        self.current_timestep = 0.0
        self.playing = False
        self.environment = environment
        self.dataset_manager = dataset_manager
        # 保存初始仿真状态，点击 Reset 时可以把整个场景恢复到启动时的状态，
        # 然后重新从数据集开头开始回放。
        self.first_state = environment.sim.clone_embodied_gaussian_state()
        self.fps = fps
    
    def reset(self):
        # 恢复初始仿真状态，重新计算 IK 相关状态，
        # 然后把数据集回放时间跳回 0。
        self.current_timestep = 0.0
        self.environment.sim.copy_embodied_gaussian_state(self.first_state)
        self.environment.sim.eval_ik()
        self.go_to_timestep(0.0)
    
    def go_to_timestep(self, timestep: float):
        # 从数据集中取出这个时刻对应的机器人关节状态 q，
        # 并把它设置成仿真器里的目标关节位置。
        #
        # 同时，用相同的时间戳刷新所有离线相机帧，
        # 这样“机器人姿态”和“视频观测”就是同步的。
        self.current_timestep = timestep
        q = self.dataset_manager.panda_state(timestep)["sheep"]["q"]
        q= torch.tensor(q, device=self.environment.sim.device)
        self.environment.set_robot_desired_q(0, q)
        self.dataset_manager.update_frames(timestep)

    def draw(self):
        # 用 ImGui 画一个简单的播放控制面板。
        imgui.begin("Playback")
        imgui.text(f"Current timestep: {self.current_timestep:.2f}")
        imgui.text(f"Playing: {self.playing}")
        _, self.fps = imgui.slider_int("FPS", self.fps, 1, 120)
        if imgui.button("Play"):
            self.playing = True
        imgui.same_line()
        if imgui.button("Pause"):
            self.playing = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.end()
    
    async def run_physics(self):
        # 按仿真器自己的固定 dt 持续推进物理。
        # 这个循环和下面的播放循环是分开的：
        # 播放循环负责改“目标关节”和“输入视频帧”，
        # 这个循环负责真正执行 physics step 和 visual forces 更新。
        dt = self.environment.dt()
        while True:
            self.environment.step()
            await trio.sleep(dt)
    
    async def run(self):
        # 并发运行“物理循环”和“播放循环”。
        # 物理一直在走；只有 playing=True 时，数据集时间才会继续往前走。
        async with trio.open_nursery() as n:
            n.start_soon(self.run_physics)
            while True:
                if self.playing:
                    self.current_timestep += 1/self.fps
                    self.go_to_timestep(self.current_timestep)
                await trio.sleep(1/self.fps)

async def main(dataset_path: Path):
    # 构建这个示例使用的 PushT 场景。
    # 这里面包含机器人、物体、Gaussian 表示以及仿真配置。
    environment = build_environment()
    # 增大 visual force 的迭代次数，让离线回放时的视觉校正更明显一点。
    environment.visual_forces_settings.iterations = 7

    # 这里加载离线数据集。
    #
    # 默认读取仓库自带的：
    # examples/embodied_environments/pusht_embodied/sample_demos/0
    #
    # 这个目录至少需要包含：
    # 1. robots.json        机器人状态时间序列
    # 2. cameras.json       相机外参 + 视频/元数据路径
    # 3. videos/*.mp4       每个相机的视频
    # 4. videos/*.json      每个相机对应的内参、分辨率、时间戳
    #
    # 如果你要换成别的数据集，现在不需要再改源码，
    # 只需要：
    # 1. 传命令行参数：python examples/example_embodied_pusht_offline.py --dataset /你的数据集目录
    # 2. 或设置环境变量：EMBODIED_GAUSSIANS_DATASET=/你的数据集目录
    #
    # 前提仍然是：你的数据集目录结构能被 DatasetManager 直接读取。
    dataset_manager = DatasetManager(dataset_path)
    # 把离线相机帧挂到 environment 上。
    # 这样仿真器在 step 时，就能把当前渲染结果和离线观测图像做比较，
    # 并计算基于图像的 visual forces。
    environment.frames = dataset_manager.frames
    fps = 60
    playback_controls = PlaybackControls(environment, dataset_manager, fps)
    # 在窗口显示之前，先把机器人状态和视频帧同步到数据集的第 0 帧。
    playback_controls.reset()


    visualizer = EmbodiedGUI()
    # GUI 每一帧都会渲染 embodied environment，
    # 同时把上面定义的播放控制面板也画出来。
    visualizer.set_environment(environment)
    visualizer.callbacks_render.append(playback_controls.draw)


    async with trio.open_nursery() as n:
        # GUI 主循环和播放控制并发运行。
        # 当窗口退出时，取消后台任务，保证进程能干净结束。
        n.start_soon(playback_controls.run)
        await visualizer.run()
        n.cancel_scope.cancel()


if __name__ == "__main__":
    # 创建仿真资源之前，要先初始化 Warp。
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    print(f"[example_embodied_pusht_offline] 使用数据集目录: {dataset_path}")
    wp.config.quiet = True
    wp.init()
    trio.run(main, dataset_path)
