#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rigidly realign a processed SUPER dataset so the fitted ground plane is perpendicular to the world Z axis."
    )
    p.add_argument("--dataset", type=Path, default=Path("data/super/grasp5_processed"))
    p.add_argument("--backup", action="store_true", default=True)
    p.add_argument("--no-backup", dest="backup", action="store_false")
    p.add_argument("--camera-side-positive-z", action="store_true", default=True,
                   help="After alignment, keep the stereo cameras on the positive-Z side of the table using a proper Rx(180) rotation.")
    p.add_argument("--camera-side-negative-z", dest="camera_side_positive_z", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fit_ground_plane(dataset: Path) -> tuple[np.ndarray, float, np.ndarray, dict]:
    ground = load(dataset / "bodies/ground.json")
    points = np.asarray(ground["particles"]["means"], dtype=np.float64)
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    a, b, c = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
    normal = np.array([-a, -b, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    d = -float(c) / np.linalg.norm(np.array([-a, -b, 1.0], dtype=np.float64))
    centroid = points.mean(axis=0)
    origin = centroid - normal * (normal.dot(centroid) + d)
    residual = points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)
    meta = {
        "fit_z_ax_by_c": [float(a), float(b), float(c)],
        "normal_before": normal.tolist(),
        "plane_d_before": float(d),
        "origin_before": origin.tolist(),
        "ground_z_percentiles_before": {str(q): float(np.percentile(points[:, 2], q)) for q in [0, 1, 5, 50, 95, 99, 100]},
        "fit_residual_percentiles_before": {str(q): float(np.percentile(residual, q)) for q in [0, 1, 5, 50, 95, 99, 100]},
    }
    return normal, d, origin, meta


def rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a.dot(axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def make_transform(dataset: Path, camera_side_positive_z: bool) -> tuple[np.ndarray, dict]:
    normal, _d, origin, meta = fit_ground_plane(dataset)
    R = rotation_from_a_to_b(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    t = -R @ origin
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R
    X[:3, 3] = t

    cameras_path = dataset / "cameras.json"
    if cameras_path.exists():
        cams = load(cameras_path)
        centers = []
        for cam in cams.values():
            if "X_WC" in cam:
                centers.append((X @ np.asarray(cam["X_WC"], dtype=np.float64))[:3, 3])
        if centers:
            mean_camera_z = float(np.mean([c[2] for c in centers]))
            if camera_side_positive_z and mean_camera_z < 0.0:
                F = np.diag([1.0, -1.0, -1.0, 1.0])
                X = F @ X
            elif (not camera_side_positive_z) and mean_camera_z > 0.0:
                F = np.diag([1.0, -1.0, -1.0, 1.0])
                X = F @ X

    meta["X_realign_from_previous_world"] = X.tolist()
    meta["camera_side_positive_z"] = bool(camera_side_positive_z)
    meta["right_handed_rotation_det"] = float(np.linalg.det(X[:3, :3]))
    return X, meta


def transform_points(points, X: np.ndarray):
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return points
    out = (X[:3, :3] @ arr.reshape(-1, 3).T + X[:3, 3:4]).T
    return out.reshape(arr.shape).tolist()


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [(0.25 * s), (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def transform_quats(quats, R: np.ndarray):
    return [matrix_to_quat_wxyz(R @ quat_wxyz_to_matrix(q)) for q in quats]


def transform_xform(T, X: np.ndarray):
    return (X @ np.asarray(T, dtype=np.float64)).tolist()


def transform_body(path: Path, X: np.ndarray) -> None:
    if not path.exists():
        return
    d = load(path)
    if "X_WB" in d:
        d["X_WB"] = np.eye(4, dtype=np.float64).tolist()
    R = X[:3, :3]
    for section in ("particles", "gaussians"):
        if section not in d or d[section] is None:
            continue
        if "means" in d[section]:
            d[section]["means"] = transform_points(d[section]["means"], X)
        if "quats" in d[section]:
            d[section]["quats"] = transform_quats(d[section]["quats"], R)
    d["coordinate_frame"] = "right_handed_table_world_z_up_m"
    save(path, d)


def transform_motion_means(path: Path, X: np.ndarray) -> None:
    if not path.exists():
        return
    d = load(path)
    for state in d.get("states", []):
        if "means" in state:
            state["means"] = transform_points(state["means"], X)
    d["coordinate_frame"] = "right_handed_table_world_z_up_m"
    save(path, d)


def transform_cameras(path: Path, X: np.ndarray) -> None:
    if not path.exists():
        return
    d = load(path)
    for cam in d.values():
        if "X_WC" in cam:
            cam["X_WC"] = transform_xform(cam["X_WC"], X)
    save(path, d)


def transform_lnd_motion(path: Path, X: np.ndarray) -> None:
    if not path.exists():
        return
    d = load(path)
    for state in d.get("states", []):
        if "link_transforms_rectified_camera" in state:
            for k, T in state["link_transforms_rectified_camera"].items():
                state["link_transforms_rectified_camera"][k] = transform_xform(T, X)
        if "keypoints_rectified_camera" in state:
            for k, v in state["keypoints_rectified_camera"].items():
                state["keypoints_rectified_camera"][k] = transform_points([v], X)[0]
    d["coordinate_frame"] = "right_handed_table_world_z_up_m"
    save(path, d)


def transform_generic_xforms(path: Path, X: np.ndarray) -> None:
    if not path.exists():
        return
    d = load(path)
    for k, v in list(d.items()):
        if isinstance(v, list) and len(v) == 4 and all(isinstance(row, list) and len(row) == 4 for row in v):
            if k.startswith("T_") or k.startswith("X_"):
                d[k] = transform_xform(v, X)
    d["world_frame"] = "right_handed_table_world_z_up_m"
    save(path, d)


def validate(dataset: Path) -> dict:
    points = np.asarray(load(dataset / "bodies/ground.json")["particles"]["means"], dtype=np.float64)
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    a, b, c = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
    normal = np.array([-a, -b, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    angle = math.degrees(math.acos(np.clip(abs(normal.dot(np.array([0.0, 0.0, 1.0]))), -1.0, 1.0)))
    cams = load(dataset / "cameras.json") if (dataset / "cameras.json").exists() else {}
    camera_centers = {k: np.asarray(v["X_WC"], dtype=np.float64)[:3, 3].tolist() for k, v in cams.items() if "X_WC" in v}
    return {
        "post_fit_z_ax_by_c": [float(a), float(b), float(c)],
        "post_normal": normal.tolist(),
        "post_angle_to_z_deg": float(angle),
        "post_ground_z_percentiles": {str(q): float(np.percentile(points[:, 2], q)) for q in [0, 1, 5, 50, 95, 99, 100]},
        "post_camera_centers": camera_centers,
    }


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    X, meta = make_transform(dataset, args.camera_side_positive_z)
    meta["validation_before"] = validate(dataset)
    print(json.dumps({"transform": meta, "validation_before": meta["validation_before"]}, indent=2))
    if args.dry_run:
        return
    if args.backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = dataset.parent / f"{dataset.name}_before_z_axis_realign_{stamp}"
        shutil.copytree(dataset, backup)
        print(f"backup: {backup}")

    for rel in ["bodies/tissue.json", "bodies/ground.json", "bodies/instrument_tip.json"]:
        transform_body(dataset / rel, X)
        print(f"transformed {rel}")
    transform_motion_means(dataset / "bodies/instrument_tip_motion.json", X)
    print("transformed bodies/instrument_tip_motion.json")
    transform_cameras(dataset / "cameras.json", X)
    print("transformed cameras.json")
    for rel in ["instruments/psm1_lnd_model.json", "instruments/urdf/psm1_urdf_articulation.json"]:
        transform_generic_xforms(dataset / rel, X)
    transform_lnd_motion(dataset / "instruments/psm1_lnd_motion.json", X)
    save(dataset / "bodies/ground_plane.json", {"plane": [0.0, 0.0, 1.0, 0.0], "coordinate_frame": "right_handed_table_world_z_up_m"})

    meta["validation_after"] = validate(dataset)
    meta["created_at"] = datetime.now().isoformat()
    save(dataset / "super_ground_z_axis_realign.json", meta)
    print(json.dumps(meta["validation_after"], indent=2))


if __name__ == "__main__":
    main()
