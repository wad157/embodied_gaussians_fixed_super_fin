#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Build SUPER tissue/ground bodies from manual masks and depth.')
    repo = Path(__file__).resolve().parents[1]
    p.add_argument('--repo', type=Path, default=repo)
    p.add_argument('--rgb', type=Path, default=repo / 'data/super/grasp5_native/rgb/000000-left-540.png')
    p.add_argument('--depth', type=Path, default=Path('/home/hsieh/data0/wad/depth/000000.npy'))
    p.add_argument('--calib', type=Path, default=repo / 'data/super/grasp5_native/calib_rectified.json')
    p.add_argument('--tissue-mask', type=Path, default=repo / 'data/super/grasp5_native/masks/000000-tissue.png')
    p.add_argument('--ground-mask', type=Path, default=repo / 'data/super/grasp5_native/masks/000000-ground.png')
    p.add_argument('--out-bodies', type=Path, default=repo / 'data/super/grasp5_offline_demo/bodies')
    p.add_argument('--out-objects', type=Path, default=repo / 'examples/embodied_environments/super_embodied/objects')
    p.add_argument('--out-env', type=Path, default=repo / 'examples/embodied_environments/super_embodied/environment')
    p.add_argument('--debug', type=Path, default=repo / 'data/super/grasp5_offline_demo/body_debug')
    p.add_argument('--max-points', type=int, default=2000)
    p.add_argument('--particle-radius', type=float, default=0.003)
    p.add_argument('--gaussian-scale', type=float, default=0.003)
    p.add_argument('--opacity', type=float, default=0.5)
    p.add_argument('--min-depth', type=float, default=0.04)
    p.add_argument('--max-depth', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=17)
    return p.parse_args()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f'failed to read mask: {path}')
    if m.shape != shape:
        raise ValueError(f'mask shape {m.shape} != expected {shape}: {path}')
    return m > 0


def k_for_image(calib_path: Path, shape: tuple[int, int]) -> np.ndarray:
    calib = load_json(calib_path)
    K = np.asarray(calib.get('K_rectified', calib.get('K_left_rect')), dtype=np.float64).copy()
    h, w = shape
    src_size = calib.get('rectified_size') or calib.get('original_size') or [1920, 1080]
    src_w, src_h = float(src_size[0]), float(src_size[1])
    sx, sy = w / src_w, h / src_h
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def backproject(mask: np.ndarray, depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, min_depth: float, max_depth: float):
    valid = mask & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points = np.column_stack([x, y, z])
    colors = rgb[ys, xs].astype(np.float64) / 255.0
    pixels = np.column_stack([xs, ys])
    return points, colors, pixels


def fit_plane_svd(points: np.ndarray, trim_iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        raise ValueError('need at least 3 points to fit plane')
    keep = np.ones(len(points), dtype=bool)
    plane = None
    residual = None
    for _ in range(trim_iters):
        pts = points[keep]
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        n = vh[-1].astype(np.float64)
        n /= np.linalg.norm(n) + 1e-12
        if n[2] < 0:
            n *= -1.0
        d = -float(n @ centroid)
        plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
        residual = points @ n + d
        abs_res = np.abs(residual[keep])
        med = float(np.median(abs_res))
        mad = float(np.median(np.abs(abs_res - med))) + 1e-9
        threshold = max(0.0025, med + 3.0 * 1.4826 * mad)
        new_keep = np.abs(residual) <= threshold
        if new_keep.sum() < 100:
            break
        keep = new_keep
    assert plane is not None and residual is not None
    return plane, residual


def stratified_sample(points: np.ndarray, colors: np.ndarray, pixels: np.ndarray, max_points: int, seed: int):
    if len(points) <= max_points:
        return points, colors, pixels
    rng = np.random.default_rng(seed)
    grid = np.floor(pixels / 24).astype(np.int32)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    bins = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
    rng.shuffle(bins)
    selected: list[int] = []
    per_bin = max(1, max_points // max(1, len(bins)))
    for b in bins:
        take = min(per_bin, len(b), max_points - len(selected))
        if take > 0:
            selected.extend(rng.choice(b, size=take, replace=False).tolist())
        if len(selected) >= max_points:
            break
    if len(selected) < max_points:
        remaining = np.setdiff1d(np.arange(len(points)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        take = min(max_points - len(selected), len(remaining))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    idx = np.asarray(selected, dtype=np.int64)
    return points[idx], colors[idx], pixels[idx]


def make_body(name: str, world_points: np.ndarray, colors: np.ndarray, particle_radius: float, gaussian_scale: float, opacity: float, body_id=None):
    center = world_points.mean(axis=0)
    local = world_points - center[None, :]
    X_WB = np.eye(4, dtype=np.float64)
    X_WB[:3, 3] = center
    quats = [[1.0, 0.0, 0.0, 0.0]] * len(local)
    point_list = local.astype(float).tolist()
    color_list = np.clip(colors, 0.0, 1.0).astype(float).tolist()
    body = {
        'name': name,
        'X_WB': X_WB.astype(float).tolist(),
        'particles': {
            'means': point_list,
            'quats': quats,
            'radii': [float(particle_radius)] * len(local),
            'colors': color_list,
        },
        'gaussians': {
            'means': point_list,
            'quats': quats,
            'scales': [[float(gaussian_scale)] * 3 for _ in range(len(local))],
            'opacities': [float(opacity)] * len(local),
            'colors': color_list,
        },
    }
    if body_id is not None:
        body['body_id'] = int(body_id)
    return body


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def overlay_points(path: Path, rgb: np.ndarray, tissue_mask: np.ndarray, ground_mask: np.ndarray, tissue_pix: np.ndarray, ground_pix: np.ndarray) -> None:
    out = rgb.copy()
    out[tissue_mask] = (0.55 * out[tissue_mask] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    out[ground_mask] = (0.65 * out[ground_mask] + 0.35 * np.array([40, 255, 70])).astype(np.uint8)
    for x, y in tissue_pix[::max(1, len(tissue_pix)//1000)]:
        cv2.circle(out, (int(x), int(y)), 1, (255, 255, 255), -1)
    for x, y in ground_pix[::max(1, len(ground_pix)//1000)]:
        cv2.circle(out, (int(x), int(y)), 1, (0, 0, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f'failed to read rgb: {args.rgb}')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f'depth shape {depth.shape} != rgb shape {rgb.shape[:2]}')
    K = k_for_image(args.calib, depth.shape)
    tissue_mask = load_mask(args.tissue_mask, depth.shape)
    ground_mask = load_mask(args.ground_mask, depth.shape)

    tissue_points, tissue_colors, tissue_pixels = backproject(tissue_mask, depth, rgb, K, args.min_depth, args.max_depth)
    ground_points, ground_colors, ground_pixels = backproject(ground_mask, depth, rgb, K, args.min_depth, args.max_depth)
    if len(tissue_points) < 20:
        raise ValueError(f'not enough tissue points: {len(tissue_points)}')
    if len(ground_points) < 20:
        raise ValueError(f'not enough ground points: {len(ground_points)}')

    plane, residual = fit_plane_svd(ground_points)
    tissue_points_s, tissue_colors_s, tissue_pixels_s = stratified_sample(tissue_points, tissue_colors, tissue_pixels, args.max_points, args.seed)
    ground_points_s, ground_colors_s, ground_pixels_s = stratified_sample(ground_points, ground_colors, ground_pixels, args.max_points, args.seed + 1)

    tissue_body = make_body('tissue', tissue_points_s, tissue_colors_s, args.particle_radius, args.gaussian_scale, args.opacity)
    ground_body = make_body('ground', ground_points_s, ground_colors_s, args.particle_radius, args.gaussian_scale, args.opacity, body_id=-1)
    plane_json = {'plane': plane.astype(float).tolist()}

    for out in [args.out_bodies, args.out_objects]:
        save_json(out / 'tissue.json', tissue_body)
        save_json(out / 'ground.json', ground_body)
    save_json(args.out_bodies / 'ground_plane.json', plane_json)
    save_json(args.out_env / 'ground_plane.json', plane_json)

    overlay_points(args.debug / '000000_manual_masks_points_overlay.png', rgb, tissue_mask, ground_mask, tissue_pixels_s, ground_pixels_s)
    np.save(args.debug / 'tissue_points_camera.npy', tissue_points_s.astype(np.float32))
    np.save(args.debug / 'ground_points_camera.npy', ground_points_s.astype(np.float32))

    summary = {
        'frame': 0,
        'coordinate_frame': 'rectified_left_camera_opencv_m',
        'K_used': K.astype(float).tolist(),
        'depth_path': str(args.depth),
        'rgb_path': str(args.rgb),
        'tissue_mask_path': str(args.tissue_mask),
        'ground_mask_path': str(args.ground_mask),
        'tissue_mask_pixels': int(tissue_mask.sum()),
        'ground_mask_pixels': int(ground_mask.sum()),
        'raw_tissue_depth_points': int(len(tissue_points)),
        'raw_ground_depth_points': int(len(ground_points)),
        'sampled_tissue_points': int(len(tissue_points_s)),
        'sampled_ground_points': int(len(ground_points_s)),
        'ground_plane': plane.astype(float).tolist(),
        'ground_plane_residual_m_percentiles': np.percentile(residual, [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist(),
        'abs_ground_plane_residual_m_percentiles': np.percentile(np.abs(residual), [50, 90, 95, 99]).astype(float).tolist(),
        'tissue_world_bounds_min': tissue_points_s.min(axis=0).astype(float).tolist(),
        'tissue_world_bounds_max': tissue_points_s.max(axis=0).astype(float).tolist(),
        'ground_world_bounds_min': ground_points_s.min(axis=0).astype(float).tolist(),
        'ground_world_bounds_max': ground_points_s.max(axis=0).astype(float).tolist(),
        'tissue_X_WB_translation': np.asarray(tissue_body['X_WB'])[:3, 3].astype(float).tolist(),
        'ground_X_WB_translation': np.asarray(ground_body['X_WB'])[:3, 3].astype(float).tolist(),
    }
    save_json(args.debug / 'manual_body_build_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
