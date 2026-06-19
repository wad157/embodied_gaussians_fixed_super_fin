#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'data/super/grasp5_offline_demo'
BODIES = DATASET / 'bodies'
OBJECTS = REPO / 'examples/embodied_environments/super_embodied/objects'
DEBUG = DATASET / 'body_debug'
RGB = REPO / 'data/super/grasp5_native/rgb/000000-left-540.png'
CALIB = REPO / 'data/super/grasp5_native/calib_rectified.json'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def camera_view_rays_to_z0(xs: np.ndarray, ys: np.ndarray, X_WC: np.ndarray, K: np.ndarray) -> np.ndarray:
    C = X_WC[:3, 3]
    R = X_WC[:3, :3]
    dirs_c = np.column_stack([
        (xs - K[0, 2]) / K[0, 0],
        (ys - K[1, 2]) / K[1, 1],
        np.ones_like(xs, dtype=np.float64),
    ])
    dirs_w = (R @ dirs_c.T).T
    denom = dirs_w[:, 2]
    lam = -C[2] / denom
    pts = C[None, :] + lam[:, None] * dirs_w
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(lam) & (lam > 0) & (np.abs(denom) > 1e-9)
    return pts, valid


def body_world_points(body: dict) -> np.ndarray:
    local = np.asarray(body['particles']['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    return (X[:3, :3] @ local.T + X[:3, 3:4]).T


def make_body(points_w: np.ndarray, colors: np.ndarray):
    center = points_w.mean(axis=0)
    local = points_w - center[None, :]
    X = np.eye(4, dtype=np.float64)
    X[:3, 3] = center
    quats = [[1.0, 0.0, 0.0, 0.0]] * len(points_w)
    local_list = local.astype(float).tolist()
    color_list = np.clip(colors, 0.0, 1.0).astype(float).tolist()
    return {
        'name': 'ground',
        'body_id': -1,
        'X_WB': X.astype(float).tolist(),
        'coordinate_frame': 'right_handed_table_world_z_up_m',
        'sampling': 'uniform_image_view_rays_intersect_z0_table_plane',
        'particles': {
            'means': local_list,
            'quats': quats,
            'radii': [0.003] * len(points_w),
            'colors': color_list,
        },
        'gaussians': {
            'means': local_list,
            'quats': quats,
            'scales': [[0.003, 0.003, 0.003] for _ in range(len(points_w))],
            'opacities': [0.5] * len(points_w),
            'colors': color_list,
        },
    }


def project_world(points_w: np.ndarray, X_WC: np.ndarray, K: np.ndarray):
    X_CW = np.linalg.inv(X_WC)
    pts_c = (X_CW[:3, :3] @ points_w.T + X_CW[:3, 3:4]).T
    z = pts_c[:, 2]
    u = K[0, 0] * pts_c[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_c[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def main():
    DEBUG.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.imread(str(RGB), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f'missing rgb: {RGB}')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    K = np.asarray(load(CALIB)['K_left_rect'], dtype=np.float64)
    cams = load(DATASET / 'cameras.json')
    X_WC = np.asarray(cams['stereo_left']['X_WC'], dtype=np.float64)

    target = 2000
    # Image-uniform sampling. Use centers of a near-regular grid over the whole image.
    nx = 50
    ny = 40
    xs = np.linspace(0.5, w - 1.5, nx)
    ys = np.linspace(0.5, h - 1.5, ny)
    xx, yy = np.meshgrid(xs, ys)
    pix = np.column_stack([xx.ravel(), yy.ravel()])
    pts, valid = camera_view_rays_to_z0(pix[:, 0], pix[:, 1], X_WC, K)
    pts = pts[valid]
    pix = pix[valid]
    pts[:, 2] = 0.0
    if len(pts) != target:
        print(f'warning: valid points {len(pts)} != target {target}')

    ui = np.clip(np.rint(pix[:, 0]).astype(np.int32), 0, w - 1)
    vi = np.clip(np.rint(pix[:, 1]).astype(np.int32), 0, h - 1)
    sampled = rgb[vi, ui].astype(np.float64) / 255.0
    neutral = np.array([0.20, 0.12, 0.10], dtype=np.float64)
    colors = np.clip(0.65 * sampled + 0.35 * neutral, 0.04, 1.0)

    body = make_body(pts, colors)
    save(BODIES / 'ground.json', body)
    save(OBJECTS / 'ground.json', body)

    # Debug overlay that should visibly cover the whole camera view.
    tissue = load(BODIES / 'tissue.json')
    tissue_pts = body_world_points(tissue)
    uv_t, z_t = project_world(tissue_pts, X_WC, K)
    uv_g, z_g = project_world(pts, X_WC, K)
    overlay = rgb.copy()
    for u, v in uv_g:
        cv2.circle(overlay, (int(round(u)), int(round(v))), 2, (40, 255, 70), -1, cv2.LINE_AA)
    mt = (z_t > 0) & (uv_t[:, 0] >= 0) & (uv_t[:, 0] < w) & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < h)
    for u, v in uv_t[mt][::max(1, mt.sum() // 1600)]:
        cv2.circle(overlay, (int(round(u)), int(round(v))), 1, (255, 40, 40), -1, cv2.LINE_AA)
    cv2.imwrite(str(DEBUG / 'ground_camera_view_full_coverage_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    summary = {
        'method': 'uniform image pixels -> camera rays -> z=0 table-world plane',
        'ground_points': int(len(pts)),
        'pixel_grid': [nx, ny],
        'z_percentiles': np.percentile(pts[:, 2], [0, 1, 50, 99, 100]).astype(float).tolist(),
        'xy_bounds_min': pts[:, :2].min(axis=0).astype(float).tolist(),
        'xy_bounds_max': pts[:, :2].max(axis=0).astype(float).tolist(),
        'projected_uv_bounds': [float(uv_g[:,0].min()), float(uv_g[:,1].min()), float(uv_g[:,0].max()), float(uv_g[:,1].max())],
        'debug_overlay': str(DEBUG / 'ground_camera_view_full_coverage_overlay.png'),
    }
    save(DEBUG / 'ground_camera_view_full_coverage_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
