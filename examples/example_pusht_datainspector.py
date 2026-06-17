# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path

import marsoom
import marsoom.cuda
import trio
import tyro
import warp as wp
from marsoom import imgui
from trio_util import periodic

from embodied_gaussians.physics_simulator.loader import Loader
from sim_environments.pusht import PushTEnvironment
from embodied_gaussians.vis import SimulationViewer

@dataclass
class Settings:
    path: tyro.conf.PositionalRequiredArgs[Path]
    """数据集根目录。该目录下应包含按数字编号的多个 demo 子目录。"""

class InspectGUI(marsoom.Window):
    def __init__(self, settings: Settings):
        super().__init__(caption="Inspector")
        self.settings = settings
        
        # 初始化一个 PushT 仿真环境，用来承载离线数据回放。
        # 注意这里不是“重新跑控制策略”，而是把已存好的 state/control
        # 一帧一帧灌回仿真器里显示出来。
        self.pusht_env = PushTEnvironment.build()
        self.loader = Loader(device="cuda")
        self.sim_renderer = SimulationViewer(self)
        self.sim_renderer.set_simulator(self.pusht_env.simulator())
        
        # 记录当前播放进度和播放状态。
        self.time_index = 0
        self.playing = False
        self.play_nursery = None
        self.nursery = None
        
        # 扫描数据集目录下的所有 demo 子目录。
        # 这里默认每个子目录名都是数字，比如 0/、1/、2/。
        self.demo_paths = sorted([d for d in self.settings.path.iterdir() if d.is_dir()], 
                               key=lambda x: int(x.name))
        self.demo_number = 0
        if self.demo_paths:
            self.load_demo(0)

    def load_demo(self, index: int):
        # 切换到指定编号的 demo：
        # 1. 重置 Loader
        # 2. 从磁盘加载该 demo
        # 3. 跳到第 0 帧
        print(f"Loading demo {index}")
        self.demo_number = index
        self.loader.reset()
        self.loader.load(self.demo_paths[index])
        self.time_index = 0
        self.go_to_index(self.time_index)

    def go_to_index(self, index: int):
        # 从离线文件中取出某一帧对应的：
        # - 仿真状态 state
        # - 控制量 control
        # 然后直接写回当前仿真器。
        #
        # 所以这个文件非常适合用来检查“你整理出来的数据集能不能被 Loader 正确读回”。
        state, control = self.loader.get_state_at_index(index)
        self.pusht_env.simulator().set_state(state)
        self.pusht_env.simulator().set_control(control)

    async def start_playing(self):
        assert not self.playing
        self.playing = True
        async with trio.open_nursery() as self.play_nursery:
            async for _ in periodic(1 / 60.0):
                # 按固定帧率递增 time_index，实现离线回放。
                self.time_index += 1
                self.time_index %= self.loader.num_steps
                self.go_to_index(self.time_index)

    def stop_playing(self):
        if self.play_nursery is None:
            return
        self.playing = False
        self.play_nursery.cancel_scope.cancel()

    async def run(self):
        async def render_loop():
            render_fps = 60
            async for _ in periodic(1 / render_fps):
                if self.has_exit:
                    break
                self.step()

        async with trio.open_nursery() as self.nursery:
            await render_loop()
            self.nursery.cancel_scope.cancel()

    def keyboard(self):
        # 空格键切换播放/暂停。
        if imgui.is_key_pressed(imgui.Key.space):
            if self.playing:
                self.stop_playing()
            else:
                self.nursery.start_soon(self.start_playing)

    def render(self):
        self.keyboard()
        
        # Main control window with better styling
        imgui.set_next_window_size((400, 300), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.no_collapse)
        
        # Status information
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
        imgui.begin_child("Status", (0, 100), True)
        imgui.text("Playback Controls")
        imgui.separator()
        
        c, self.time_index = imgui.slider_int(
            "Time", self.time_index, 0, self.loader.num_steps - 1
        )
        if c:
            # 手动拖动进度条时，立即跳到对应帧。
            self.time_index = min(max(0, self.time_index), self.loader.num_steps - 1)
            self.go_to_index(self.time_index)

        button_size = (imgui.get_content_region_avail()[0] / 2 - 5, 30)
        if not self.playing:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.6, 0.2, 1.0))
            if imgui.button("Play", button_size):
                self.nursery.start_soon(self.start_playing)
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.2, 0.2, 1.0))
            if imgui.button("Stop", button_size):
                self.stop_playing()
            imgui.pop_style_color()
            
        imgui.end_child()
        imgui.pop_style_color()

        imgui.spacing()
        imgui.text(f"Demo number: {self.demo_number}")
        imgui.separator()

        # 左侧 demo 列表用于切换不同轨迹。
        imgui.begin_child("Demos", (0, 0), True)
        flags = imgui.SelectableFlags_.allow_double_click.value
        for i, path in enumerate(self.demo_paths):
            selected = i == self.demo_number
            clicked, _selected = imgui.selectable(f"Demo {path.name}", selected, flags=flags)
            if clicked and not selected:
                self.load_demo(i)
        imgui.end_child()
        
        imgui.end()

        # 3D 窗口只负责展示当前离线帧对应的仿真状态。
        imgui.begin("3D Viewer")
        with self.sim_renderer.draw(True):
            self.sim_renderer.render_meshes()
        self.sim_renderer.process_nav()
        imgui.end()

async def main():
    # 这个示例的定位是“检查已经保存好的 PushT 数据集”。
    # 如果你在整理别的数据集时，不确定 Loader 是否还能正确读取，
    # 这个文件通常是第一批要跟着一起改和一起验证的。
    wp.init()
    settings = tyro.cli(Settings)
    window = InspectGUI(settings)
    await window.run()

if __name__ == "__main__":
    trio.run(main)
