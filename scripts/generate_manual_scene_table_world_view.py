#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'data/super/grasp5_offline_demo'
OUT_DIR = DATASET / 'body_debug'


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def body_world_points(body: dict, section='particles') -> np.ndarray:
    local = np.asarray(body[section]['means'], dtype=np.float64)
    X = np.asarray(body.get('X_WB', np.eye(4)), dtype=np.float64)
    return (X[:3, :3] @ local.T + X[:3, 3:4]).T


def camera_center(cam: dict) -> np.ndarray:
    return np.asarray(cam['X_WC'], dtype=np.float64)[:3, 3]


def draw_camera(ax, X_WC: np.ndarray, label: str, color: str):
    C = X_WC[:3, 3]
    R = X_WC[:3, :3]
    # Small camera axes/frustum in world coordinates. OpenCV camera looks along +Z.
    scale = 0.018
    axes = {
        'x': (R[:, 0], 'r'),
        'y': (R[:, 1], 'g'),
        'z': (R[:, 2], 'b'),
    }
    for _, (vec, c) in axes.items():
        P = C + scale * vec
        ax.plot([C[0], P[0]], [C[1], P[1]], [C[2], P[2]], color=c, linewidth=2)
    z = R[:, 2]
    x = R[:, 0]
    y = R[:, 1]
    center = C + scale * 1.6 * z
    corners = [
        center + scale * 0.8 * x + scale * 0.45 * y,
        center - scale * 0.8 * x + scale * 0.45 * y,
        center - scale * 0.8 * x - scale * 0.45 * y,
        center + scale * 0.8 * x - scale * 0.45 * y,
    ]
    for P in corners:
        ax.plot([C[0], P[0]], [C[1], P[1]], [C[2], P[2]], color=color, linewidth=0.8)
    cyc = corners + [corners[0]]
    ax.plot([p[0] for p in cyc], [p[1] for p in cyc], [p[2] for p in cyc], color=color, linewidth=1.2)
    ax.scatter([C[0]], [C[1]], [C[2]], color=color, s=50, marker='^')
    ax.text(C[0], C[1], C[2] + 0.006, label, color=color, fontsize=9)


def set_equal_axes(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.06)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.25, center[2] + radius * 0.9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tissue = load(DATASET / 'bodies/tissue.json')
    ground = load(DATASET / 'bodies/ground.json')
    plane = load(DATASET / 'bodies/ground_plane.json')
    cams = load(DATASET / 'cameras.json')

    tissue_pts = body_world_points(tissue)
    ground_pts = body_world_points(ground)
    cam_pts = np.vstack([camera_center(cams['stereo_left']), camera_center(cams['stereo_right'])])
    all_pts = np.vstack([tissue_pts, ground_pts, cam_pts])

    summary = {
        'frame': 'right_handed_table_world_z_up_m',
        'ground_plane': plane,
        'tissue_points': int(len(tissue_pts)),
        'ground_points': int(len(ground_pts)),
        'tissue_bounds_min': tissue_pts.min(axis=0).astype(float).tolist(),
        'tissue_bounds_max': tissue_pts.max(axis=0).astype(float).tolist(),
        'ground_bounds_min': ground_pts.min(axis=0).astype(float).tolist(),
        'ground_bounds_max': ground_pts.max(axis=0).astype(float).tolist(),
        'camera_centers': {k: camera_center(v).astype(float).tolist() for k, v in cams.items()},
        'camera_height_above_plane_m': {k: float(camera_center(v)[2]) for k, v in cams.items()},
        'tissue_z_percentiles_m': {str(q): float(np.percentile(tissue_pts[:, 2], q)) for q in [0, 1, 5, 50, 95, 99, 100]},
        'ground_z_percentiles_m': {str(q): float(np.percentile(ground_pts[:, 2], q)) for q in [0, 1, 5, 50, 95, 99, 100]},
    }
    with open(OUT_DIR / 'table_world_scene_view_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(gs[0, :], projection='3d'), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    ax = axes[0]
    ax.scatter(ground_pts[:, 0], ground_pts[:, 1], ground_pts[:, 2], s=4, c='#2ca25f', alpha=0.55, label='ground points')
    ax.scatter(tissue_pts[:, 0], tissue_pts[:, 1], tissue_pts[:, 2], s=5, c='#de2d26', alpha=0.75, label='tissue points')
    # z=0 plane patch covering data extent.
    x_min, y_min = all_pts[:, :2].min(axis=0) - 0.015
    x_max, y_max = all_pts[:, :2].max(axis=0) + 0.015
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='#bdbdbd', alpha=0.18, linewidth=0, shade=False)
    draw_camera(ax, np.asarray(cams['stereo_left']['X_WC'], dtype=np.float64), 'left cam', '#3182bd')
    draw_camera(ax, np.asarray(cams['stereo_right']['X_WC'], dtype=np.float64), 'right cam', '#08519c')
    ax.set_title('Right-handed table world: cameras, table plane, tissue, ground')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z up (m)')
    ax.legend(loc='upper left')
    set_equal_axes(ax, all_pts)
    ax.view_init(elev=22, azim=-60)

    views = [
        ('Top view: X-Y', 0, 1, 'X (m)', 'Y (m)'),
        ('Side view: X-Z', 0, 2, 'X (m)', 'Z (m)'),
        ('Side view: Y-Z', 1, 2, 'Y (m)', 'Z (m)'),
    ]
    for ax2, (title, a, b, xlabel, ylabel) in zip(axes[1:], views):
        ax2.scatter(ground_pts[:, a], ground_pts[:, b], s=3, c='#2ca25f', alpha=0.45)
        ax2.scatter(tissue_pts[:, a], tissue_pts[:, b], s=4, c='#de2d26', alpha=0.65)
        for name, cam in cams.items():
            C = camera_center(cam)
            ax2.scatter([C[a]], [C[b]], s=60, marker='^', label=name)
            ax2.text(C[a], C[b], name, fontsize=8)
        if b == 2:
            ax2.axhline(0.0, color='black', linewidth=1.2, alpha=0.8)
        if a == 2:
            ax2.axvline(0.0, color='black', linewidth=1.2, alpha=0.8)
        ax2.set_title(title)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.25)
        ax2.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    out = OUT_DIR / 'table_world_scene_view.png'
    fig.savefig(out, dpi=180)
    print(out)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
