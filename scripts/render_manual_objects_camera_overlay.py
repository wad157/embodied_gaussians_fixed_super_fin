#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'data/super/grasp5_offline_demo'
NATIVE = REPO / 'data/super/grasp5_native'
OUT = DATASET / 'body_debug'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def body_world_points(body: dict, section='particles') -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(body[section]['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    pts = (X[:3, :3] @ local.T + X[:3, 3:4]).T
    colors = np.asarray(body[section].get('colors', [[1, 1, 1]] * len(local)), dtype=np.float64)
    return pts, colors


def project_world_to_camera(points_w: np.ndarray, X_WC: np.ndarray, K: np.ndarray):
    X_CW = np.linalg.inv(X_WC)
    pts_c = (X_CW[:3, :3] @ points_w.T + X_CW[:3, 3:4]).T
    z = pts_c[:, 2]
    u = K[0, 0] * pts_c[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_c[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z, pts_c


def draw_points(img, uv, z, color_bgr, radius=2):
    out = img.copy()
    h, w = out.shape[:2]
    mask = np.isfinite(z) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.flatnonzero(mask)
    # far first, near last
    idx = idx[np.argsort(z[idx])[::-1]]
    for i in idx:
        cv2.circle(out, (int(round(uv[i, 0])), int(round(uv[i, 1]))), radius, color_bgr, -1, cv2.LINE_AA)
    return out, int(len(idx)), int(mask.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb_left = cv2.imread(str(NATIVE / 'rgb/000000-left-540.png'), cv2.IMREAD_COLOR)
    if rgb_left is None:
        raise RuntimeError('missing 000000-left-540.png')
    h, w = rgb_left.shape[:2]
    calib = load(NATIVE / 'calib_rectified.json')
    # calib_rectified currently stores K_left_rect for the 540p working image
    # and K_rectified for the original 1080p image. Prefer the matrix whose
    # focal length is already consistent with the loaded image size.
    K_left = np.asarray(calib.get('K_left_rect', calib.get('K_rectified')), dtype=np.float64).copy()
    K_full = np.asarray(calib.get('K_rectified', K_left), dtype=np.float64).copy()
    if K_left[0, 0] < 0.75 * w and K_left[1, 1] < 2.0 * h:
        K = K_left
    else:
        src_size = calib.get('rectified_size') or [1920, 1080]
        sx, sy = w / float(src_size[0]), h / float(src_size[1])
        K = K_full
        K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy

    tissue = load(DATASET / 'bodies/tissue.json')
    ground = load(DATASET / 'bodies/ground.json')
    cams = load(DATASET / 'cameras.json')
    X_WC = np.asarray(cams['stereo_left']['X_WC'], dtype=np.float64)

    tissue_pts, _ = body_world_points(tissue)
    ground_pts, _ = body_world_points(ground)
    uv_t, z_t, _ = project_world_to_camera(tissue_pts, X_WC, K)
    uv_g, z_g, _ = project_world_to_camera(ground_pts, X_WC, K)

    overlay = rgb_left.copy()
    overlay, gt_drawn, gt_visible = draw_points(overlay, uv_g, z_g, (60, 220, 60), radius=1)
    overlay, ti_drawn, ti_visible = draw_points(overlay, uv_t, z_t, (40, 40, 255), radius=1)

    blended = cv2.addWeighted(overlay, 0.72, rgb_left, 0.28, 0.0)
    cv2.putText(blended, 'camera view overlay: tissue=red ground=green', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(blended, 'camera view overlay: tissue=red ground=green', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    out_path = OUT / 'camera_view_left_overlay_540.png'
    cv2.imwrite(str(out_path), blended)

    # Also save a cleaner two-panel image: original | overlay.
    sep = np.full((h, 8, 3), 255, dtype=np.uint8)
    left_label = rgb_left.copy()
    cv2.putText(left_label, 'original left camera', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(left_label, 'original left camera', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    contact = np.concatenate([left_label, sep, blended], axis=1)
    contact_path = OUT / 'camera_view_left_original_vs_overlay_540.png'
    cv2.imwrite(str(contact_path), contact)

    summary = {
        'image': str(NATIVE / 'rgb/000000-left-540.png'),
        'output_overlay': str(out_path),
        'output_original_vs_overlay': str(contact_path),
        'K_used': K.astype(float).tolist(),
        'tissue_visible_points': ti_visible,
        'ground_visible_points': gt_visible,
        'tissue_total_points': int(len(tissue_pts)),
        'ground_total_points': int(len(ground_pts)),
        'tissue_uv_bounds': [uv_t[np.isfinite(z_t) & (z_t > 0), 0].min().item(), uv_t[np.isfinite(z_t) & (z_t > 0), 1].min().item(), uv_t[np.isfinite(z_t) & (z_t > 0), 0].max().item(), uv_t[np.isfinite(z_t) & (z_t > 0), 1].max().item()],
        'ground_uv_bounds': [uv_g[np.isfinite(z_g) & (z_g > 0), 0].min().item(), uv_g[np.isfinite(z_g) & (z_g > 0), 1].min().item(), uv_g[np.isfinite(z_g) & (z_g > 0), 0].max().item(), uv_g[np.isfinite(z_g) & (z_g > 0), 1].max().item()],
    }
    with open(OUT / 'camera_view_left_overlay_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
