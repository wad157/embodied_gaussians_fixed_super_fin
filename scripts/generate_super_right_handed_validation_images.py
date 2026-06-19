#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D_BLENDER_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def points(body: dict, section: str = "particles") -> np.ndarray:
    return np.asarray(body[section]["means"], dtype=np.float64)


def camera_opencv_pose(camera: dict) -> np.ndarray:
    return np.asarray(camera["X_WC"], dtype=np.float64) @ D_BLENDER_OPENCV


def project(pts_world: np.ndarray, X_WC_opencv: np.ndarray, K: np.ndarray):
    ph = np.c_[pts_world, np.ones(len(pts_world))].T
    pts_cam = (np.linalg.inv(X_WC_opencv) @ ph)[:3].T
    uvw = (K @ pts_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, pts_cam[:, 2]


def draw_points(img: np.ndarray, uv: np.ndarray, depth: np.ndarray, color, radius=2) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    mask = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for u, v in uv[mask].astype(int):
        cv2.circle(out, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def equal_axes(ax, arrays):
    all_pts = np.concatenate(arrays, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    span = max(span, 0.12)
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], centers):
        setter(c - span / 2, c + span / 2)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "data/super/grasp5_native"
    tissue = load_json(repo / "examples/embodied_environments/super_embodied/objects/tissue.json")
    ground = load_json(repo / "examples/embodied_environments/super_embodied/objects/ground.json")
    cameras = load_json(repo / "data/super/grasp5_offline_demo/cameras.json")
    left_md = load_json(repo / "data/super/grasp5_offline_demo/videos/stereo_left.json")
    right_md = load_json(repo / "data/super/grasp5_offline_demo/videos/stereo_right.json")
    K_left = np.asarray(left_md["K"], dtype=np.float64)
    K_right = np.asarray(right_md["K"], dtype=np.float64)

    tissue_pts = points(tissue)
    ground_pts = points(ground)
    cam_centers = np.array([camera_opencv_pose(c)[:3, 3] for c in cameras.values()])

    fig = plt.figure(figsize=(15, 5))
    views = [(25, -55, "Perspective"), (0, -90, "XZ"), (0, 0, "YZ")]
    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(ground_pts[:, 0], ground_pts[:, 1], ground_pts[:, 2], s=2, c="#2563eb", label="ground")
        ax.scatter(tissue_pts[:, 0], tissue_pts[:, 1], tissue_pts[:, 2], s=3, c="#dc2626", label="tissue")
        ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], s=45, c="#111827", marker="^", label="cameras")
        ax.set_title(title)
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
        ax.set_zlabel("z m")
        ax.view_init(elev=elev, azim=azim)
        equal_axes(ax, [ground_pts, tissue_pts, cam_centers])
        if i == 1:
            ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "scene_3d_right_handed.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name, K in [(axes[0], "stereo_left", K_left), (axes[1], "stereo_right", K_right)]:
        X = camera_opencv_pose(cameras[name])
        uv_g, z_g = project(ground_pts, X, K)
        uv_t, z_t = project(tissue_pts, X, K)
        ax.scatter(uv_g[z_g > 0, 0], uv_g[z_g > 0, 1], s=1, c="#2563eb", label="ground")
        ax.scatter(uv_t[z_t > 0, 0], uv_t[z_t > 0, 1], s=2, c="#dc2626", label="tissue")
        ax.set_title(f"{name}: right-handed table projection")
        ax.set_xlim(0, left_md["resolution"][0])
        ax.set_ylim(left_md["resolution"][1], 0)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "projection_right_handed.png", dpi=180)
    plt.close(fig)

    left_rgb = cv2.imread(str(repo / "data/super/grasp5_native/rgb/000000-left.png"), cv2.IMREAD_COLOR)
    right_rgb = cv2.imread(str(repo / "data/super/grasp5_native/rgb/000000-right.png"), cv2.IMREAD_COLOR)
    if left_rgb is None or right_rgb is None:
        raise FileNotFoundError("Could not load frame 000000 RGB images")

    overlays = []
    for img, name, K in [(left_rgb, "stereo_left", K_left), (right_rgb, "stereo_right", K_right)]:
        X = camera_opencv_pose(cameras[name])
        uv_g, z_g = project(ground_pts, X, K)
        uv_t, z_t = project(tissue_pts, X, K)
        overlay = draw_points(img, uv_g, z_g, (255, 80, 20), radius=1)
        overlay = draw_points(overlay, uv_t, z_t, (20, 20, 255), radius=2)
        cv2.putText(overlay, name, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        overlays.append(overlay)
    combined = np.concatenate(overlays, axis=1)
    cv2.imwrite(str(out_dir / "projection_overlay_right_handed.png"), combined)

    print("wrote:")
    for name in ["scene_3d_right_handed.png", "projection_right_handed.png", "projection_overlay_right_handed.png"]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
