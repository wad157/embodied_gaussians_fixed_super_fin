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


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def body_world_points(body: dict) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(body['particles']['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    pts = (X[:3, :3] @ local.T + X[:3, 3:4]).T
    colors = np.asarray(body['particles'].get('colors', [[0.2,0.12,0.1]] * len(local)), dtype=np.float64)
    return pts, colors


def project_world(points_w: np.ndarray, X_WC: np.ndarray, K: np.ndarray):
    X_CW = np.linalg.inv(X_WC)
    pts_c = (X_CW[:3,:3] @ points_w.T + X_CW[:3,3:4]).T
    z = pts_c[:,2]
    u = K[0,0] * pts_c[:,0] / z + K[0,2]
    v = K[1,1] * pts_c[:,1] / z + K[1,2]
    return np.column_stack([u,v]), z


def unproject_z0_pixels(xs: np.ndarray, ys: np.ndarray, X_WC: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Camera ray in camera coords: [(u-cx)/fx, (v-cy)/fy, 1]. Solve C + lambda*d_world intersects world z=0.
    C = X_WC[:3,3]
    R = X_WC[:3,:3]
    dirs_c = np.column_stack([(xs - K[0,2]) / K[0,0], (ys - K[1,2]) / K[1,1], np.ones_like(xs, dtype=np.float64)])
    dirs_w = (R @ dirs_c.T).T
    lam = -C[2] / dirs_w[:,2]
    pts = C[None,:] + lam[:,None] * dirs_w
    return pts


def make_body(points_w: np.ndarray, colors: np.ndarray):
    center = points_w.mean(axis=0)
    local = points_w - center[None,:]
    X = np.eye(4, dtype=np.float64)
    X[:3,3] = center
    quats = [[1.0,0.0,0.0,0.0]] * len(points_w)
    color_list = np.clip(colors,0,1).astype(float).tolist()
    local_list = local.astype(float).tolist()
    return {
        'name': 'ground',
        'body_id': -1,
        'X_WB': X.astype(float).tolist(),
        'coordinate_frame': 'right_handed_table_world_z_up_m',
        'sampling': 'uniform_grid_on_z0_table_plane_including_under_tissue',
        'particles': {
            'means': local_list,
            'quats': quats,
            'radii': [0.003] * len(points_w),
            'colors': color_list,
        },
        'gaussians': {
            'means': local_list,
            'quats': quats,
            'scales': [[0.003,0.003,0.003] for _ in range(len(points_w))],
            'opacities': [0.5] * len(points_w),
            'colors': color_list,
        },
    }


def main():
    DEBUG.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(RGB), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f'missing rgb {RGB}')
    h,w = rgb.shape[:2]
    rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    tissue = load(BODIES / 'tissue.json')
    cams = load(DATASET / 'cameras.json')
    cam = np.asarray(cams['stereo_left']['X_WC'], dtype=np.float64)
    calib = load(REPO / 'data/super/grasp5_native/calib_rectified.json')
    K = np.asarray(calib['K_left_rect'], dtype=np.float64)

    tissue_pts, _ = body_world_points(tissue)
    tissue_xy = tissue_pts[:,:2]

    # Determine visible z=0 table rectangle from image corners, then clamp to a practical region around tissue.
    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float64)
    corner_pts = unproject_z0_pixels(corners[:,0], corners[:,1], cam, K)
    img_min = corner_pts[:,:2].min(axis=0)
    img_max = corner_pts[:,:2].max(axis=0)
    tissue_min = tissue_xy.min(axis=0)
    tissue_max = tissue_xy.max(axis=0)
    margin = np.array([0.035, 0.035], dtype=np.float64)
    xy_min = np.maximum(img_min, tissue_min - margin)
    xy_max = np.minimum(img_max, tissue_max + margin)
    # If clamp is too tight due to perspective, fall back to tissue bounds plus margin.
    if np.any(xy_max <= xy_min):
        xy_min = tissue_min - margin
        xy_max = tissue_max + margin

    target = 2000
    nx = 70
    ny = 55
    xs = np.linspace(xy_min[0], xy_max[0], nx)
    ys = np.linspace(xy_min[1], xy_max[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float64)])

    # Keep points visible in left image.
    uv, z = project_world(pts, cam, K)
    visible = (z > 0) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)

    # Keep the full visible table grid, including points underneath the tissue.
    # The table plane is a physical/support plane; it should not have a hole
    # where tissue sits. Rendering/occlusion should be handled downstream.
    keep = visible
    pts = pts[keep]
    uv = uv[keep]

    # If more than target, uniformly subsample in grid order to retain coverage.
    if len(pts) > target:
        idx = np.linspace(0, len(pts)-1, target).round().astype(np.int64)
        pts = pts[idx]
        uv = uv[idx]
    if len(pts) < 200:
        raise RuntimeError(f'too few ground grid points after filtering: {len(pts)}')

    ui = np.clip(np.rint(uv[:,0]).astype(np.int32), 0, w-1)
    vi = np.clip(np.rint(uv[:,1]).astype(np.int32), 0, h-1)
    sampled = rgb_rgb[vi, ui].astype(np.float64) / 255.0
    neutral = np.array([0.20, 0.12, 0.10], dtype=np.float64)
    colors = np.clip(0.65 * sampled + 0.35 * neutral, 0.04, 1.0)

    body = make_body(pts, colors)
    save(BODIES / 'ground.json', body)
    save(OBJECTS / 'ground.json', body)

    # Debug overlay: tissue red points and uniform ground green points.
    overlay = rgb_rgb.copy()
    uv_t, z_t = project_world(tissue_pts, cam, K)
    mt = (z_t > 0) & (uv_t[:,0] >= 0) & (uv_t[:,0] < w) & (uv_t[:,1] >= 0) & (uv_t[:,1] < h)
    for u,v in uv_t[mt][::max(1, mt.sum()//1500)]:
        cv2.circle(overlay, (int(round(u)), int(round(v))), 1, (255,40,40), -1)
    for u,v in uv[::max(1, len(uv)//1800)]:
        cv2.circle(overlay, (int(round(u)), int(round(v))), 1, (40,255,70), -1)
    cv2.imwrite(str(DEBUG / 'uniform_ground_grid_camera_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    summary = {
        'ground_points': int(len(pts)),
        'xy_min': xy_min.astype(float).tolist(),
        'xy_max': xy_max.astype(float).tolist(),
        'tissue_bounds_xy_min': tissue_min.astype(float).tolist(),
        'tissue_bounds_xy_max': tissue_max.astype(float).tolist(),
        'image_z0_bounds_xy_min': img_min.astype(float).tolist(),
        'image_z0_bounds_xy_max': img_max.astype(float).tolist(),
        'z_percentiles': np.percentile(pts[:,2], [0,1,50,99,100]).astype(float).tolist(),
        'overlay': str(DEBUG / 'uniform_ground_grid_camera_overlay.png'),
    }
    save(DEBUG / 'uniform_ground_grid_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
