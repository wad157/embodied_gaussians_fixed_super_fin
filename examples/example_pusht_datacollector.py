# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass, field
from pathlib import Path

import marsoom
import marsoom.cuda
import torch
import trio
import tyro
import warp as wp
from marsoom import guizmo, imgui
from trio_util import periodic


from sim_environments.pusht import PushTEnvironment, PushTEnvironmentActions
from helpers.canvas_2d import Canvas2D
from embodied_gaussians import PhysicsSettings, Saver
from embodied_gaussians.vis import SimulationViewer


@dataclass
class Settings:
    path: tyro.conf.PositionalRequiredArgs[Path]
    """数据集保存根目录。每次保存会在这个目录下创建一个新的数字子目录。"""
    physics: PhysicsSettings = field(default_factory=lambda: PhysicsSettings())
    """物理参数配置。"""


class CollectGUI(marsoom.Window):
    def __init__(self, settings: Settings):
        super().__init__(caption="PushT Data Collector")
        self.settings = settings

        # 检查已有 demo 的编号。
        # 保存时每个 demo 都会被写成一个独立目录，例如：
        # path/
        #   0/
        #   1/
        #   2/
        #
        # 这里通过现有目录名推断下一个可用编号。
        self.demo_number = 0
        if self.settings.path.exists():
            dirs = [d for d in self.settings.path.iterdir() if d.is_dir()]
            dirs = sorted(dirs, key=lambda x: int(x.name))
            self.demo_number = int(dirs[-1].name) + 1

        # 这个环境是“纯 PushT 仿真环境”，不是 embodied-gaussian 离线回放环境。
        # 它负责真实地产生状态轨迹，供后面的 Saver 写盘。
        self.pusht_env = PushTEnvironment.build()
        self.sim_renderer = SimulationViewer(self)
        self.sim_renderer.set_simulator(self.pusht_env.simulator())
        self.actions = PushTEnvironmentActions.allocate(1, device="cuda")
        # Saver 是离线数据写盘的核心组件。
        # 它会把每一帧的 state / control 记录下来，最后保存成一个 demo 目录。
        self.saver = Saver(self.pusht_env.simulator(), device="cuda")
        self.viewer_2d = Canvas2D(self)
        self.target_transform = wp.transformf((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.recording = True
        self.auto_reset = True
        self.press_latch = False
        self.new_demo()

    async def run(self):
        async def physics_loop():
            dt = self.pusht_env.dt()
            async for _ in periodic(dt):
                # 每个物理步：
                # 1. 推进一步仿真
                # 2. 如果开启 recording，就把当前状态和控制量写进缓存
                self.pusht_env.step()
                if self.recording:
                    self.saver.record_state_and_control()
                    self.update()

        async def render_loop():
            render_fps = 60
            async for _ in periodic(1 / render_fps):
                if self.has_exit:
                    break
                self.step()

        async with trio.open_nursery() as n:
            n.start_soon(physics_loop)
            await render_loop()
            n.cancel_scope.cancel()

    def draw_2d_viewer(self):
        # 2D 平面控制视图：
        # - 绿色 T-block：目标位姿
        # - 默认颜色 T-block：当前物体位姿
        # - 绿色圆点：期望控制位置
        # - 蓝色圆点：当前 pusher 位置
        imgui.begin("Control Plane")
        self.viewer_2d.draw()
        self.viewer_2d.draw_tblock(
            wp.transform_get_translation(self.target_transform),
            wp.transform_get_rotation(self.target_transform),
            (0.0, 1.0, 0.0, 0.7),
        )
        obs = self.pusht_env.observe()
        current_transform = obs.tblock_transforms[0].cpu().numpy()
        joint_q = obs.pusher_positions[0].cpu().numpy()
        control = self.actions.pusher_desired_positions[0].cpu().numpy()
        self.viewer_2d.draw_tblock(current_transform[:3], current_transform[3:])
        hovered = self.viewer_2d.circle(
            position=control[:2],  # type: ignore
            color=(0, 1, 0, 1),
            radius=0.015,
            thickness=2,
            unit=marsoom.eViewerUnit.UNIT,
        )
        self.viewer_2d.circle(
            position=joint_q[:2],  # type: ignore
            color=(0, 0, 1, 1),
            radius=0.02,
            thickness=2,
            unit=marsoom.eViewerUnit.UNIT,
        )
        if hovered or self.press_latch:
            # 鼠标按住或按空格时，拖动控制 pusher 的目标位置。
            if imgui.is_mouse_down(0) or imgui.is_key_down(imgui.Key.space):
                self.press_latch = True
                x, y = self.viewer_2d.get_mouse_position(unit=marsoom.eViewerUnit.UNIT)
                self.set_joints(x, y)
            else:
                self.press_latch = False

        imgui.end()

    def reset(self):
        # 清掉当前录制缓存，并重置环境。
        self.pusht_env.reset()
        self.saver.clear_allocation()
        self.press_latch = False

    def new_demo(self):
        # 开始录制一个新的 demo，但暂时不落盘。
        self.recording = True
        self.saver.clear_allocation()

    def save_demo(self):
        # 把当前缓存写到 path/<demo_number>/ 目录。
        # 这一步生成的数据结构，就是 datainspector 后面要读取的结构。
        self.saver.save(self.settings.path / f"{self.demo_number}")
        self.new_demo()
        self.demo_number += 1

    def keyboard(self):
        if imgui.is_key_pressed(imgui.Key.q):
            if self.recording:
                self.reset()

    def update(self):
        if self.auto_reset:
            # 一旦任务完成：
            # 1. 保存这一条 demo
            # 2. 重置环境，继续录制下一条
            done = self.pusht_env.done()[0].cpu().numpy()
            if done:
                self.save_demo()
                self.reset()

    def set_joints(self, x, y):
        # 这里把 2D 平面上的目标点写入动作，
        # 再调用 env.act 把控制量送进仿真器。
        self.actions.pusher_desired_positions[0].copy_(torch.tensor([x, y], device="cuda"))
        self.pusht_env.act(self.actions)

    def render(self):
        self.keyboard()
        
        # Main control window with better styling and organization
        imgui.set_next_window_size((400, 300), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.no_collapse)
        
        # Status information in a colored frame
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
        imgui.begin_child("Status", (0, 100), True)
        imgui.text("Status Information")
        imgui.separator()
        t = self.pusht_env.simulator().get_time()
        imgui.text(f"Simulation Time: {t:.2f}s")
        imgui.text(f"Recording Buffer: {self.saver.current_index}/{self.saver.max_frames}")
        imgui.text(f"Current Demo: #{self.demo_number}")
        imgui.text(f"Environment State: {'Done' if self.pusht_env.done()[0].cpu().numpy() else 'Active'}")
        imgui.end_child()
        imgui.pop_style_color()

        imgui.spacing()
        
        # Control buttons in a grid layout
        button_size = (imgui.get_content_region_avail()[0] / 2 - 5, 30)
        
        # Save Demo button with color
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.6, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.7, 0.3, 1.0))
        if imgui.button("Save Demo##save", button_size):
            self.save_demo()
        imgui.pop_style_color(2)
        
        imgui.same_line()
        
        # Reset button with color
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.2, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.8, 0.3, 0.3, 1.0))
        if imgui.button("Reset##reset", button_size):
            self.reset()
        imgui.pop_style_color(2)

        imgui.spacing()
        imgui.spacing()

        # Settings section
        imgui.text("Settings")
        imgui.separator()
        
        # Auto-reset checkbox with better styling
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.3, 0.3, 0.3, 1.0))
        _, self.auto_reset = imgui.checkbox("Auto Reset on Completion", self.auto_reset)
        imgui.pop_style_color(2)

        # Help tooltip
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically saves and resets the demo when completed")

        # Keyboard Shortcuts section
        imgui.spacing()
        imgui.spacing()
        imgui.text("Keyboard Shortcuts")
        imgui.separator()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.8, 0.8, 0.8, 1.0))
        imgui.text("Q - Reset current recording")
        imgui.text("Space - Hold to control pusher position")
        imgui.pop_style_color()

        imgui.end()

        # 下面两个窗口只是用来辅助人操作和观察：
        # - 2D Viewer 方便拖动控制
        # - 3D Viewer 方便看完整仿真状态
        imgui.set_next_window_size((600, 400), cond=imgui.Cond_.first_use_ever)
        self.draw_2d_viewer()
        
        imgui.set_next_window_size((600, 400), cond=imgui.Cond_.first_use_ever)
        imgui.begin("3D Viewer", flags=imgui.WindowFlags_.no_collapse)
        with self.sim_renderer.draw(True):
            self.sim_renderer.render_meshes()
        self.sim_renderer.process_nav()
        imgui.end()



async def main():
    # 这个示例的作用是“采集 PushT 仿真数据集”。
    # 如果你想做自己的离线数据，一般需要先理解这个文件最终写出来的目录格式。
    wp.init()
    settings = tyro.cli(Settings)
    window = CollectGUI(settings)
    await window.run()


if __name__ == "__main__":
    trio.run(main)
