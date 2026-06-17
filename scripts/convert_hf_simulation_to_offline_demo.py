#!/usr/bin/env python3

import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np
import zarr

import warp as wp
import torch

from embodied_gaussians.utils.physics_utils import save_builder, transform_from_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
SRC_DIR = REPO_ROOT / "src"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face embodied_gaussians simulation scene into the offline demo format used by example_embodied_pusht_offline.py."
    )
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Path to a simulation directory containing transforms.json, robot.json, and color/*.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "temp" / "converted_hf" / "fall_1_demo",
        help="Output directory for the converted offline demo.",
    )
    parser.add_argument(
        "--robot-name",
        default="sheep",
        help="Robot name to write into robots.json. Keep the default if you want compatibility with example_embodied_pusht_offline.py.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of creating symlinks.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_camera_manifest(scene_dir: Path, output_dir: Path, copy_files: bool) -> None:
    transforms = json.loads((scene_dir / "transforms.json").read_text())
    timestamps = transforms["timestamps"]
    cameras = transforms["cameras"]

    videos_dir = output_dir / "videos"
    ensure_dir(videos_dir)

    manifest: dict[str, dict[str, object]] = {}

    for index, camera in enumerate(cameras):
        # simulated/fall_1 没有 serial，这里优先用 name；如果没有，就退化成 camera_<index>
        camera_name = str(camera.get("name") or f"camera_{index}")
        src_video = scene_dir / camera["color_path"]
        dst_video = videos_dir / f"{camera_name}{src_video.suffix}"
        link_or_copy(src_video, dst_video, copy_files)

        metadata_rel = Path("videos") / f"{camera_name}.json"
        metadata_path = output_dir / metadata_rel
        metadata = {
            "serial": camera_name,
            "K": [
                [camera["fl_x"], 0.0, camera["cx"]],
                [0.0, camera["fl_y"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            "resolution": [camera["w"], camera["h"]],
            "timestamps": timestamps,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        manifest[camera_name] = {
            "X_WC": camera["transform_matrix"],
            "video_path": str(Path("videos") / dst_video.name),
            "metadata_path": str(metadata_rel),
        }

    (output_dir / "cameras.json").write_text(json.dumps(manifest, indent=2))


def build_robot_manifest(scene_dir: Path, output_dir: Path, robot_name: str) -> None:
    robot = json.loads((scene_dir / "robot.json").read_text())
    timestamps = robot["timestamps"]
    joints = robot["joints"]

    # example_embodied_pusht_offline.py 当前从 ["sheep"]["q"] 取关节，并且场景是 Panda，
    # 所以这里要求每帧至少提供 7 维关节。
    states = [{"q": joint[:7]} for joint in joints]
    controls = [joint[:7] for joint in joints]

    manifest = {
        robot_name: {
            "control": controls,
            "control_timestamps": timestamps,
            "states": states,
            "states_timestamps": timestamps,
        }
    }
    (output_dir / "robots.json").write_text(json.dumps(manifest, indent=2))


def build_physics_zarr(scene_dir: Path, output_dir: Path) -> None:
    # 这一步尽量少改源码：直接把 fall_1 的 body_eval 序列转成
    # 当前 Loader 能读的 physics.zarr 格式。
    #
    # 这里需要一个 builder.pckl。我们不重建完整仿真，只复用当前仓库里
    # PushT embodied 的 builder 结构，然后把 TBlock 的 body_q 按帧写进去。
    #
    # 注意：原示例里真正用到的是 DatasetManager 的时间轴和 body_q，
    # 所以只写这些最小字段即可。
    from embodied_environments.pusht_embodied.pusht_embodied import build_environment, BODY_ID

    env = build_environment()
    builder = env.sim.builder

    body_count = builder.body_count
    if body_count <= 0:
        raise RuntimeError("No body found in current builder")

    # fall_1 的 bodies_eval.json 里是逐帧的 TBlock 世界坐标点云。
    eval_data = json.loads((scene_dir.parent / "bodies_eval.json").read_text())
    eval_body = eval_data["bodies"][0]
    frames = np.array(eval_body["p_WPs"], dtype=np.float32)
    timestamps = np.array(eval_data["timestamps"], dtype=np.float32)

    # bodies_eval 本身就是同一组点的逐帧刚体轨迹。
    # 所以直接用第 0 帧做模板即可，后续每一帧都拟合到这个模板上。
    template = frames[0]

    def rigid_fit(A: np.ndarray, B: np.ndarray):
        ac = A.mean(axis=0)
        bc = B.mean(axis=0)
        A0 = A - ac
        B0 = B - bc
        H = A0.T @ B0
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # fall_1 的点云是按列向量变换来写的，所以这里要用列向量约定：
        #   B ~= R @ A + t
        # 这样才能和 bodies_eval.json 对齐。
        t = bc - R @ ac
        return R, t

    # 先把初始 state 作为模板，保证机器人/地面等其他 body 不被破坏。
    # 再逐帧覆盖 TBlock 的 body pose。
    robot_q0 = np.array(json.loads((scene_dir / "robot.json").read_text())["joints"][0][:7], dtype=np.float32)
    env.sim.set_articulation_q(0, torch.tensor(robot_q0, device=env.sim.device))
    base_body_q = wp.to_torch(env.sim.state_0.body_q).cpu().numpy()
    base_body_qd = wp.to_torch(env.sim.state_0.body_qd).cpu().numpy()
    base_body_f = wp.to_torch(env.sim.state_0.body_f).cpu().numpy()

    body_q = np.repeat(base_body_q[None, :, :], len(frames), axis=0).astype(np.float32)
    body_qd = np.zeros((len(frames), body_count, 6), dtype=np.float32)
    body_f = np.zeros((len(frames), body_count, 6), dtype=np.float32)
    body_qd[:] = base_body_qd[None, :, :]
    body_f[:] = base_body_f[None, :, :]

    robot_data = json.loads((scene_dir / "robot.json").read_text())
    joint_q = np.array([j[:7] for j in robot_data["joints"]], dtype=np.float32)
    joint_qd = np.zeros((len(frames), len(builder.joint_act)), dtype=np.float32)
    joint_qd = np.zeros((len(frames), len(builder.joint_act)), dtype=np.float32)
    control = joint_q.copy()

    target_body_id = int(BODY_ID)
    if target_body_id >= body_count:
        raise RuntimeError(f"TBlock body id {target_body_id} out of range for body_count={body_count}")

    for i, frame in enumerate(frames):
        env.sim.set_articulation_q(0, torch.tensor(joint_q[i], device=env.sim.device))
        body_q[i] = wp.to_torch(env.sim.state_0.body_q).cpu().numpy()
        R, t = rigid_fit(template, frame)
        X = np.eye(4, dtype=np.float32)
        X[:3, :3] = R.astype(np.float32)
        X[:3, 3] = t.astype(np.float32)
        body_q[i, target_body_id] = transform_from_matrix(X)

    root = zarr.open_group(output_dir / "physics.zarr", mode="w")
    root.create_array("timestamps", data=timestamps, dtype="f4")
    root.create_array("state_body_q", data=body_q, dtype="f4")
    root.create_array("state_body_qd", data=body_qd, dtype="f4")
    root.create_array("state_body_f", data=body_f, dtype="f4")
    root.create_array("state_joint_q", data=joint_q, dtype="f4")
    root.create_array("state_joint_qd", data=joint_qd, dtype="f4")
    root.create_array("control_joint_act", data=control, dtype="f4")
    save_builder(output_dir / "builder.pckl", builder)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not (scene_dir / "transforms.json").exists():
        raise FileNotFoundError(f"Missing transforms.json under {scene_dir}")
    if not (scene_dir / "robot.json").exists():
        raise FileNotFoundError(f"Missing robot.json under {scene_dir}")

    ensure_dir(output_dir)
    build_camera_manifest(scene_dir, output_dir, args.copy_videos)
    build_robot_manifest(scene_dir, output_dir, args.robot_name)
    build_physics_zarr(scene_dir, output_dir)

    print(output_dir)


if __name__ == "__main__":
    main()
