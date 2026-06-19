#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'data/super/grasp5_offline_demo'
BODIES = DATASET / 'bodies'
OBJECTS = REPO / 'examples/embodied_environments/super_embodied/objects'
ENV = REPO / 'examples/embodied_environments/super_embodied/environment'
DEBUG = DATASET / 'body_debug'
CALIB = REPO / 'data/super/grasp5_native/calib_rectified.json'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v) + 1e-12
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1].astype(np.float64)
    n /= np.linalg.norm(n) + 1e-12
    if n[2] < 0:
        n *= -1.0
    d = -float(n @ centroid)
    residual = points @ n + d
    return np.r_[n, d], residual, centroid


def body_world_points(body: dict) -> np.ndarray:
    local = np.asarray(body['particles']['means'], dtype=np.float64)
    X_WB = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    return (X_WB[:3, :3] @ local.T + X_WB[:3, 3:4]).T


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
        q = [0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        q = [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q.tolist()


def transform_body_preserve_center(body: dict, X: np.ndarray) -> dict:
    out = json.loads(json.dumps(body))
    X_old = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    # The body stores points in local coordinates relative to X_WB.
    # Apply the world-frame transform to X_WB only; rotating local means too
    # would rotate every point twice when world points are reconstructed.
    X_new = X @ X_old
    out['X_WB'] = X_new.astype(float).tolist()
    out['coordinate_frame'] = 'right_handed_table_world_z_up_m'
    return out


def baseline_from_calib(calib: dict) -> float:
    if 'baseline_m' in calib:
        return abs(float(calib['baseline_m']))
    if 'P2' in calib:
        P2 = np.asarray(calib['P2'], dtype=np.float64)
        K = np.asarray(calib.get('K_rectified', calib.get('K_left_rect')), dtype=np.float64)
        return abs(float(P2[0, 3] / K[0, 0]))
    if 'T_stereo_m' in calib:
        return float(np.linalg.norm(np.asarray(calib['T_stereo_m'], dtype=np.float64).reshape(-1)[:3]))
    return 0.005306


def make_cameras(X: np.ndarray) -> dict:
    existing = load(DATASET / 'cameras.json') if (DATASET / 'cameras.json').exists() else {}
    calib = load(CALIB)
    baseline = baseline_from_calib(calib)
    left = np.eye(4, dtype=np.float64)
    # In the rectified pair, right camera center is +baseline along left camera X in the table-frame convention used before.
    right = np.eye(4, dtype=np.float64)
    right[:3, 3] = [baseline, 0.0, 0.0]
    cams = {
        'stereo_left': dict(existing.get('stereo_left', {})),
        'stereo_right': dict(existing.get('stereo_right', {})),
    }
    cams['stereo_left']['X_WC'] = (X @ left).astype(float).tolist()
    cams['stereo_right']['X_WC'] = (X @ right).astype(float).tolist()
    cams['stereo_left']['coordinate_frame'] = 'right_handed_table_world_z_up_m'
    cams['stereo_right']['coordinate_frame'] = 'right_handed_table_world_z_up_m'
    cams['stereo_left']['baseline_source_m'] = baseline
    cams['stereo_right']['baseline_source_m'] = baseline
    return cams


def validate_body_plane(body: dict):
    pts = body_world_points(body)
    plane, residual, _ = fit_plane(pts)
    angle = math.degrees(math.acos(np.clip(abs(float(plane[:3] @ np.array([0.0,0.0,1.0]))), -1.0, 1.0)))
    return {
        'plane': plane.astype(float).tolist(),
        'angle_to_z_deg': float(angle),
        'z_percentiles': {str(q): float(np.percentile(pts[:,2], q)) for q in [0,1,5,50,95,99,100]},
        'abs_residual_percentiles_m': {str(q): float(np.percentile(np.abs(residual), q)) for q in [50,90,95,99,100]},
        'bounds_min': pts.min(axis=0).astype(float).tolist(),
        'bounds_max': pts.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    tissue = load(BODIES / 'tissue.json')
    ground = load(BODIES / 'ground.json')
    ground_pts = body_world_points(ground)
    plane, residual_before, centroid = fit_plane(ground_pts)
    normal, d = plane[:3], float(plane[3])
    origin = centroid - normal * (normal @ centroid + d)
    R_align = rotation_from_a_to_b(normal, np.array([0.0, 0.0, 1.0]))
    X_align = np.eye(4, dtype=np.float64)
    X_align[:3, :3] = R_align
    X_align[:3, 3] = -R_align @ origin

    # Keep cameras/tissue above the table with a proper rotation, not a reflection.
    left_after = X_align @ np.eye(4)
    if left_after[2, 3] < 0.0:
        Rx180 = np.diag([1.0, -1.0, -1.0, 1.0])
        X = Rx180 @ X_align
        flip_applied = True
    else:
        X = X_align
        flip_applied = False

    tissue_new = transform_body_preserve_center(tissue, X)
    ground_new = transform_body_preserve_center(ground, X)
    cams_new = make_cameras(X)
    plane_new = {'plane': [0.0, 0.0, 1.0, 0.0], 'coordinate_frame': 'right_handed_table_world_z_up_m'}

    for base in [BODIES, OBJECTS]:
        save(base / 'tissue.json', tissue_new)
        save(base / 'ground.json', ground_new)
    save(BODIES / 'ground_plane.json', plane_new)
    save(ENV / 'ground_plane.json', plane_new)
    save(DATASET / 'cameras.json', cams_new)

    summary = {
        'created_at': datetime.now().isoformat(),
        'source_frame': 'rectified_left_camera_opencv_m',
        'target_frame': 'right_handed_table_world_z_up_m',
        'ground_plane_before': plane.astype(float).tolist(),
        'ground_plane_residual_before_m_percentiles': np.percentile(residual_before, [0,1,5,50,95,99,100]).astype(float).tolist(),
        'ground_origin_before': origin.astype(float).tolist(),
        'X_right_handed_table_from_rectified_camera_opencv': X.astype(float).tolist(),
        'camera_side_positive_z': True,
        'rx180_applied': bool(flip_applied),
        'rotation_det': float(np.linalg.det(X[:3,:3])),
        'validation_ground_after': validate_body_plane(ground_new),
        'validation_tissue_after': validate_body_plane(tissue_new),
        'camera_centers_after': {k: np.asarray(v['X_WC'], dtype=np.float64)[:3,3].astype(float).tolist() for k,v in cams_new.items()},
    }
    save(DATASET / 'super_ground_z_axis_realign.json', summary)
    save(DEBUG / 'right_handed_table_realign_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
