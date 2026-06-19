#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PATHS = [
    REPO / 'data/super/grasp5_offline_demo/bodies/ground.json',
    REPO / 'examples/embodied_environments/super_embodied/objects/ground.json',
]
DEBUG = REPO / 'data/super/grasp5_offline_demo/body_debug'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def save(path: Path, data) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def body_world_points(body: dict) -> np.ndarray:
    local = np.asarray(body['particles']['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    return (X[:3, :3] @ local.T + X[:3, 3:4]).T


def set_world_points(body: dict, world_points: np.ndarray) -> dict:
    out = json.loads(json.dumps(body))
    center = world_points.mean(axis=0)
    local = world_points - center[None, :]
    X = np.eye(4, dtype=np.float64)
    X[:3, 3] = center
    out['X_WB'] = X.astype(float).tolist()
    for section in ['particles', 'gaussians']:
        if section in out and 'means' in out[section]:
            out[section]['means'] = local.astype(float).tolist()
    out['coordinate_frame'] = 'right_handed_table_world_z_up_m'
    out['flattened_to_ground_plane'] = True
    out['flattened_plane'] = [0.0, 0.0, 1.0, 0.0]
    return out


def main():
    DEBUG.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in PATHS:
        body = load(path)
        before = body_world_points(body)
        after = before.copy()
        after[:, 2] = 0.0
        new_body = set_world_points(body, after)
        save(path, new_body)
        summary[str(path)] = {
            'points': int(len(after)),
            'z_before_percentiles_m': np.percentile(before[:, 2], [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist(),
            'z_after_percentiles_m': np.percentile(after[:, 2], [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist(),
            'X_WB_after_translation': np.asarray(new_body['X_WB'], dtype=np.float64)[:3, 3].astype(float).tolist(),
        }
    with open(DEBUG / 'ground_flatten_to_z0_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
