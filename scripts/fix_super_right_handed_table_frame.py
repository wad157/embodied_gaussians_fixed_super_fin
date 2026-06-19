#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


D_CURRENT_TO_RIGHT_HANDED = np.diag([1.0, -1.0, 1.0, 1.0])
D_BLENDER_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
RX_180 = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description=(
            "Repair the SUPER final dataset from the old single-Z-flipped frame "
            "to a right-handed table frame with camera-side +Z."
        )
    )
    p.add_argument("--repo", type=Path, default=repo)
    p.add_argument(
        "--source-psm-model",
        type=Path,
        default=Path(
            "/home/hsieh/data0/wad/embodied_gaussians_fixed_super2/"
            "data/super/grasp5_processed_before_z_axis_realign_20260610_053315/"
            "instruments/psm1_lnd_model.json"
        ),
    )
    p.add_argument("--calib", type=Path, default=repo / "data/super/grasp5_native/calib_rectified.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def transform_points(points, X: np.ndarray):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return points
    out = (X[:3, :3] @ arr.reshape(-1, 3).T + X[:3, 3:4]).T
    return out.reshape(arr.shape).tolist()


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ]
    arr = np.asarray(q, dtype=np.float64)
    arr /= np.linalg.norm(arr)
    return arr.tolist()


def transform_quats_by_basis_change(quats, D3: np.ndarray):
    # Basis reflection old->new is D3. Orientation matrices must be conjugated
    # to stay proper rotations: R_new = D3 * R_old * D3.
    out = []
    for q in quats:
        R_old = quat_wxyz_to_matrix(q)
        R_new = D3 @ R_old @ D3
        out.append(matrix_to_quat_wxyz(R_new))
    return out


def transform_body(path: Path, dry_run: bool) -> dict:
    d = load(path)
    D3 = D_CURRENT_TO_RIGHT_HANDED[:3, :3]
    stats = {"path": str(path), "coordinate_frame": d.get("coordinate_frame")}
    if "X_WB" in d:
        d["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    for section in ("particles", "gaussians"):
        if section not in d or d[section] is None:
            continue
        if "means" in d[section]:
            before = np.asarray(d[section]["means"], dtype=np.float64)
            d[section]["means"] = transform_points(d[section]["means"], D_CURRENT_TO_RIGHT_HANDED)
            after = np.asarray(d[section]["means"], dtype=np.float64)
            stats[f"{section}_z_minmax"] = [float(after[:, 2].min()), float(after[:, 2].max())]
            stats[f"{section}_y_minmax_before"] = [float(before[:, 1].min()), float(before[:, 1].max())]
            stats[f"{section}_y_minmax_after"] = [float(after[:, 1].min()), float(after[:, 1].max())]
        if "quats" in d[section]:
            d[section]["quats"] = transform_quats_by_basis_change(d[section]["quats"], D3)
    d["coordinate_frame"] = "right_handed_table_world_z_up_m"
    if not dry_run:
        save(path, d)
    return stats


def update_cameras(path: Path, table_from_source_opencv: np.ndarray, baseline_m: float, dry_run: bool) -> dict:
    d = load(path)
    stats = {}
    baseline_world = table_from_source_opencv[:3, :3] @ np.array([baseline_m, 0.0, 0.0])
    left_opencv = table_from_source_opencv.copy()
    for name, cam in d.items():
        # DatasetManager/FramesBuilder expects X_WC in Blender camera basis and
        # internally converts it to OpenCV with X_WC @ D_BLENDER_OPENCV.
        x_wc_opencv = left_opencv.copy()
        if name == "stereo_right":
            # Rectified P2 has negative fx*baseline, so the right camera center is
            # +baseline along the rectified left-camera X axis.
            x_wc_opencv[:3, 3] = left_opencv[:3, 3] + baseline_world
        cam["X_WC"] = (x_wc_opencv @ D_BLENDER_OPENCV).tolist()
        R_json = np.asarray(cam["X_WC"], dtype=np.float64)[:3, :3]
        R_opencv = (np.asarray(cam["X_WC"], dtype=np.float64) @ D_BLENDER_OPENCV)[:3, :3]
        stats[name] = {
            "json_position": np.asarray(cam["X_WC"], dtype=np.float64)[:3, 3].tolist(),
            "json_det": float(np.linalg.det(R_json)),
            "opencv_det_after_frames_builder": float(np.linalg.det(R_opencv)),
        }
    stats["right_camera_baseline_world_m"] = baseline_world.tolist()
    if not dry_run:
        save(path, d)
    return stats


def euler_xyz_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    # URDF uses fixed-axis rpy. This is equivalent to scipy Rotation.as_euler("xyz").
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def update_urdf_base(path: Path, table_from_source_opencv: np.ndarray, source_psm_model: Path, dry_run: bool) -> dict:
    source = load(source_psm_model)
    T_source_psm = np.asarray(source["T_rectified_camera_psm_base"], dtype=np.float64)
    T_table_psm = table_from_source_opencv @ T_source_psm
    R = T_table_psm[:3, :3]
    if np.linalg.det(R) < 0.0:
        raise ValueError("PSM base rotation is still improper")
    rpy = euler_xyz_from_matrix(R)
    xyz = T_table_psm[:3, 3]
    text = path.read_text()
    repl = (
        f'<origin rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}" '
        f'xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"/>'
    )
    text_new, count = re.subn(
        r'<origin rpy="[^"]+" xyz="[^"]+"\s*/>',
        repl,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Could not update first URDF origin in {path}")
    if not dry_run:
        path.write_text(text_new)
    return {"xyz": xyz.tolist(), "rpy": list(rpy), "det": float(np.linalg.det(R))}


def update_realign_metadata(path: Path, table_from_source_opencv: np.ndarray, dry_run: bool) -> dict:
    d = load(path)
    d["coordinate_fix"] = {
        "fixed_at": "2026-06-18",
        "reason": "Replace old single-axis z reflection with proper Rx(180) rotation.",
        "old_final_frame_issue": "Fz=diag(1,1,-1) produced improper camera/world rotations.",
        "new_frame": "right_handed_table_world_z_up_m",
        "X_right_handed_table_from_rectified_camera_opencv": table_from_source_opencv.tolist(),
        "X_json_camera_pose_convention": "Blender camera basis; FramesBuilder converts with diag(1,-1,-1).",
        "world_transform_from_old_final_frame": D_CURRENT_TO_RIGHT_HANDED.tolist(),
    }
    d["camera_side_negative_z"] = False
    d["camera_side_positive_z"] = True
    if not dry_run:
        save(path, d)
    return d["coordinate_fix"]


def main() -> None:
    args = parse_args()
    repo = args.repo
    meta_path = repo / "data/super/grasp5_offline_demo/super_ground_z_axis_realign.json"
    meta = load(meta_path)
    x_align = np.asarray(meta["X_realign_from_previous_world"], dtype=np.float64)
    table_from_source_opencv = RX_180 @ x_align

    report = {
        "table_from_source_opencv_det": float(np.linalg.det(table_from_source_opencv[:3, :3])),
        "bodies": [],
    }
    for rel in [
        "examples/embodied_environments/super_embodied/objects/tissue.json",
        "examples/embodied_environments/super_embodied/objects/ground.json",
    ]:
        report["bodies"].append(transform_body(repo / rel, args.dry_run))

    calib = load(args.calib)
    report["cameras"] = update_cameras(
        repo / "data/super/grasp5_offline_demo/cameras.json",
        table_from_source_opencv,
        float(calib["baseline_m"]),
        args.dry_run,
    )
    report["urdf_base"] = update_urdf_base(
        repo / "data/super/psm_robot/psm.urdf",
        table_from_source_opencv,
        args.source_psm_model,
        args.dry_run,
    )
    report["metadata"] = update_realign_metadata(meta_path, table_from_source_opencv, args.dry_run)

    ground_plane = repo / "examples/embodied_environments/super_embodied/environment/ground_plane.json"
    ground = load(ground_plane)
    ground["coordinate_frame"] = "right_handed_table_world_z_up_m"
    if not args.dry_run:
        save(ground_plane, ground)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
