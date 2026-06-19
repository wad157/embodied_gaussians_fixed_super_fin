#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TISSUE_PATHS = [
    REPO / 'data/super/grasp5_offline_demo/bodies/tissue.json',
    REPO / 'examples/embodied_environments/super_embodied/objects/tissue.json',
]
DEBUG = REPO / 'data/super/grasp5_offline_demo/body_debug'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def world_points(body: dict) -> np.ndarray:
    local = np.asarray(body['particles']['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    return (X[:3, :3] @ local.T + X[:3, 3:4]).T


def compute_keep(points: np.ndarray) -> tuple[np.ndarray, dict]:
    keep_z = points[:, 2] >= 0.0
    pts = points[keep_z]
    if len(pts) < 50:
        return keep_z, {'z_removed': int((~keep_z).sum()), 'outlier_removed': 0}

    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med[None, :]), axis=0) + 1e-9
    score = np.abs((pts - med[None, :]) / (1.4826 * mad[None, :]))
    keep_mad = (score[:, 0] <= 5.0) & (score[:, 1] <= 5.0) & (score[:, 2] <= 5.0)

    # Extra radial guard in XY, still conservative.
    xy = pts[:, :2]
    center = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - center[None, :], axis=1)
    q1, q3 = np.percentile(dist, [25, 75])
    max_dist = q3 + 3.0 * (q3 - q1)
    keep_radial = dist <= max_dist

    local_keep = keep_mad & keep_radial
    keep = np.zeros(len(points), dtype=bool)
    keep[np.flatnonzero(keep_z)] = local_keep
    return keep, {
        'z_removed': int((~keep_z).sum()),
        'outlier_removed_after_z_filter': int(len(pts) - local_keep.sum()),
        'median_xyz': med.astype(float).tolist(),
        'mad_xyz': mad.astype(float).tolist(),
        'radial_max_m': float(max_dist),
    }


def filter_list_like(value, keep: np.ndarray):
    if isinstance(value, list) and len(value) == len(keep):
        return [v for v, k in zip(value, keep.tolist()) if k]
    return value


def rebuild_body(body: dict, keep: np.ndarray, pts_w: np.ndarray) -> dict:
    out = json.loads(json.dumps(body))
    pts_new = pts_w[keep]
    center = pts_new.mean(axis=0)
    local_new = pts_new - center[None, :]
    X = np.eye(4, dtype=np.float64)
    X[:3, 3] = center
    out['X_WB'] = X.astype(float).tolist()
    for section in ['particles', 'gaussians']:
        if section not in out:
            continue
        sec = out[section]
        sec['means'] = local_new.astype(float).tolist()
        for key in list(sec.keys()):
            if key == 'means':
                continue
            sec[key] = filter_list_like(sec[key], keep)
    out['coordinate_frame'] = 'right_handed_table_world_z_up_m'
    out['cleaning'] = {
        'removed_points_below_z0': True,
        'removed_robust_outliers': True,
        'points_before': int(len(pts_w)),
        'points_after': int(len(pts_new)),
    }
    return out


def main():
    DEBUG.mkdir(parents=True, exist_ok=True)
    primary = load(TISSUE_PATHS[0])
    pts_primary = world_points(primary)
    keep_primary, criteria = compute_keep(pts_primary)
    if keep_primary.sum() < 100:
        raise RuntimeError(f'too few tissue points remain: {keep_primary.sum()} / {len(keep_primary)}')

    summary = {'criteria': criteria, 'files': {}}
    for path in TISSUE_PATHS:
        body = load(path)
        pts = world_points(body)
        keep = keep_primary if len(pts) == len(keep_primary) else compute_keep(pts)[0]
        cleaned = rebuild_body(body, keep, pts)
        save(path, cleaned)
        after = world_points(cleaned)
        summary['files'][str(path)] = {
            'points_before': int(len(pts)),
            'points_after': int(len(after)),
            'removed': int(len(pts) - len(after)),
            'z_before_percentiles_m': np.percentile(pts[:, 2], [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist(),
            'z_after_percentiles_m': np.percentile(after[:, 2], [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist(),
            'bounds_after_min': after.min(axis=0).astype(float).tolist(),
            'bounds_after_max': after.max(axis=0).astype(float).tolist(),
        }
    save(DEBUG / 'tissue_cleaning_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
