#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

D_BLENDER_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def transform_from_origin(elem) -> np.ndarray:
    X = np.eye(4, dtype=np.float64)
    if elem is None:
        return X
    xyz = [float(v) for v in elem.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.attrib.get("rpy", "0 0 0").split()]
    X[:3, :3] = rpy_matrix(rpy)
    X[:3, 3] = xyz
    return X


def axis_angle(axis, q):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([
        [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
    ], dtype=np.float64)


def joint_motion(joint_type, axis, q):
    X = np.eye(4, dtype=np.float64)
    if joint_type in ("revolute", "continuous"):
        X[:3, :3] = axis_angle(axis, q)
    elif joint_type == "prismatic":
        X[:3, 3] = np.asarray(axis, dtype=np.float64) * q
    return X


def parse_urdf(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    children = {}
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        axis_elem = joint.find("axis")
        axis = [float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()] if axis_elem is not None else [0.0, 0.0, 1.0]
        info = {
            "name": joint.attrib["name"],
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent,
            "child": child,
            "axis": axis,
            "origin": transform_from_origin(joint.find("origin")),
        }
        children.setdefault(parent, []).append(info)
        joints.append(info)
    return links, children, joints


def compute_link_transforms(children, q_by_joint):
    transforms = {"world": np.eye(4, dtype=np.float64)}
    stack = ["world"]
    while stack:
        parent = stack.pop()
        for joint in children.get(parent, []):
            q = q_by_joint.get(joint["name"], 0.0)
            X = transforms[parent] @ joint["origin"] @ joint_motion(joint["type"], joint["axis"], q)
            transforms[joint["child"]] = X
            stack.append(joint["child"])
    return transforms


def visual_mesh_points(urdf_path: Path, links, link_transforms, total_points: int):
    mesh_items = []
    for lname, link in links.items():
        if lname not in link_transforms:
            continue
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None or geom.find("mesh") is None:
                continue
            mesh_file = geom.find("mesh").attrib["filename"]
            mesh_path = (urdf_path.parent / mesh_file).resolve()
            mesh = trimesh.load_mesh(mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                continue
            X_visual = transform_from_origin(visual.find("origin"))
            X_world = link_transforms[lname] @ X_visual
            area = float(mesh.area) if mesh.area > 0 else 1e-8
            mesh_items.append((lname, mesh, X_world, area))
    total_area = sum(item[3] for item in mesh_items)
    pts_all, colors_all = [], []
    link_point_counts = {}
    rng = np.random.default_rng(7)
    palette = {
        "PSM1_psm_base_link": [120, 120, 120],
        "PSM1_outer_yaw_link": [145, 145, 145],
        "PSM1_tool_main_link": [80, 120, 220],
        "PSM1_tool_wrist_link": [220, 120, 80],
        "PSM1_tool_wrist_shaft_link": [220, 160, 70],
        "PSM1_tool_wrist_sca_link": [230, 70, 70],
        "PSM1_tool_wrist_sca_shaft_link": [230, 40, 40],
        "PSM1_tool_wrist_sca_ee_link_1": [250, 20, 20],
        "PSM1_tool_wrist_sca_ee_link_2": [250, 20, 20],
    }
    for lname, mesh, X_world, area in mesh_items:
        n = max(100, int(round(total_points * area / total_area)))
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng)
        pts_h = np.c_[pts, np.ones(len(pts))].T
        pts_w = (X_world @ pts_h)[:3].T
        pts_all.append(pts_w)
        colors_all.append(np.tile(np.asarray(palette.get(lname, [160, 160, 160]), dtype=np.uint8), (len(pts_w), 1)))
        link_point_counts[lname] = link_point_counts.get(lname, 0) + int(len(pts_w))
    return np.concatenate(pts_all, axis=0), np.concatenate(colors_all, axis=0), link_point_counts


def write_ply(path: Path, pts: np.ndarray, colors: np.ndarray):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def camera_opencv_pose(camera: dict) -> np.ndarray:
    X_WC = np.asarray(camera["X_WC"], dtype=np.float64)
    if camera.get("coordinate_frame") == "right_handed_table_world_z_up_m":
        return X_WC
    return X_WC @ D_BLENDER_OPENCV


def project(pts_world, X_WC_opencv, K):
    ph = np.c_[pts_world, np.ones(len(pts_world))].T
    pts_cam = (np.linalg.inv(X_WC_opencv) @ ph)[:3].T
    uvw = (K @ pts_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, pts_cam[:, 2]


def draw_points(img, uv, depth, color, radius=1, max_points=60000):
    out = img.copy()
    h, w = out.shape[:2]
    mask = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    inds = np.flatnonzero(mask)
    if len(inds) > max_points:
        inds = inds[np.linspace(0, len(inds) - 1, max_points).astype(int)]
    for u, v in uv[inds].astype(int):
        cv2.circle(out, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)
    return out, len(inds), int(mask.sum())


def transform_source_points_to_table(points_source, T):
    arr = np.asarray(points_source, dtype=np.float64).reshape(-1, 3)
    out = (T[:3, :3] @ arr.T + T[:3, 3:4]).T
    return out



def expand_super_q_to_urdf_joints(q, mimic_cfg):
    q_by_joint = {}
    input_names = mimic_cfg["input_joint_names"]
    input_to_urdf = mimic_cfg["input_to_urdf_joint"]
    for input_name, value in zip(input_names, q):
        q_by_joint[input_to_urdf[input_name]] = float(value)
    for joint_name, spec in mimic_cfg["mimic"].items():
        source = spec["source"]
        q_by_joint[joint_name] = q_by_joint[source] * float(spec.get("multiplier", 1.0)) + float(spec.get("offset", 0.0))
    return q_by_joint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=120000)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    urdf_path = repo / "data/super/psm_robot/psm.urdf"
    robots = load_json(repo / "data/super/grasp5_offline_demo/robots.json")
    q = robots["sheep"]["states"][0]["q"]
    mimic_cfg = load_json(repo / "data/super/psm_robot/psm_mimic_map.json")
    q_by_joint = expand_super_q_to_urdf_joints(q, mimic_cfg)

    links, children, _ = parse_urdf(urdf_path)
    link_tf = compute_link_transforms(children, q_by_joint)
    pts, colors, link_point_counts = visual_mesh_points(urdf_path, links, link_tf, args.points)

    out_ply = repo / "data/super/psm_robot/psm_frame0_right_handed_dense.ply"
    write_ply(out_ply, pts, colors)

    cams = load_json(repo / "data/super/grasp5_offline_demo/cameras.json")
    left_md = load_json(repo / "data/super/grasp5_offline_demo/videos/stereo_left.json")
    right_md = load_json(repo / "data/super/grasp5_offline_demo/videos/stereo_right.json")
    K_left = np.asarray(left_md["K"], dtype=np.float64)
    K_right = np.asarray(right_md["K"], dtype=np.float64)
    imgs = {
        "stereo_left": cv2.imread(str(repo / "data/super/grasp5_native/rgb/000000-left.png"), cv2.IMREAD_COLOR),
        "stereo_right": cv2.imread(str(repo / "data/super/grasp5_native/rgb/000000-right.png"), cv2.IMREAD_COLOR),
    }

    motion_path = repo / "data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json"
    motion = load_json(motion_path)
    # Use fin-local LND keypoints. projection_space is the historically visible
    # raw_camera_tuned space, already transformed to table for this diagnostic.
    key_source = list(motion["states"][0]["keypoints_table_from_projection_space"].values())
    key_pts = np.asarray(key_source, dtype=np.float64)

    overlays = []
    stats = {}
    for name, K in [("stereo_left", K_left), ("stereo_right", K_right)]:
        img = imgs[name]
        X = camera_opencv_pose(cams[name])
        uv, depth = project(pts, X, K)
        overlay, drawn, visible = draw_points(img, uv, depth, (0, 220, 255), radius=1)
        uvk, depthk = project(key_pts, X, K)
        overlay, kdrawn, kvisible = draw_points(overlay, uvk, depthk, (0, 0, 255), radius=4, max_points=1000)
        cv2.putText(overlay, f"{name} PSM frame0", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        overlays.append(overlay)
        h, w = overlay.shape[:2]
        mask = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        stats[name] = {
            "mesh_visible": int(mask.sum()),
            "mesh_total": int(len(pts)),
            "keypoints_visible": int(kvisible),
            "keypoints_total": int(len(key_pts)),
            "u_range_visible": [float(uv[mask,0].min()) if mask.any() else None, float(uv[mask,0].max()) if mask.any() else None],
            "v_range_visible": [float(uv[mask,1].min()) if mask.any() else None, float(uv[mask,1].max()) if mask.any() else None],
        }

    out_dir = repo / "data/super/grasp5_native"
    cv2.imwrite(str(out_dir / "psm_projection_right_handed_dense.png"), np.concatenate(overlays, axis=1))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(pts)//25000)
    ax.scatter(pts[::step,0], pts[::step,1], pts[::step,2], s=0.6, c=colors[::step]/255.0)
    ax.scatter(key_pts[:,0], key_pts[:,1], key_pts[:,2], s=20, c="red")
    ax.set_xlabel("x m"); ax.set_ylabel("y m"); ax.set_zlabel("z m")
    ax.set_title("PSM frame0 right-handed dense sample")
    all_pts = np.vstack([pts, key_pts])
    center = (all_pts.min(axis=0)+all_pts.max(axis=0))/2
    span = max(float((all_pts.max(axis=0)-all_pts.min(axis=0)).max()), 0.25)
    ax.set_xlim(center[0]-span/2, center[0]+span/2)
    ax.set_ylim(center[1]-span/2, center[1]+span/2)
    ax.set_zlim(center[2]-span/2, center[2]+span/2)
    fig.tight_layout()
    fig.savefig(out_dir / "psm_scene_right_handed_dense.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"ply": str(out_ply), "points": int(len(pts)), "link_point_counts": link_point_counts, "q_by_joint": q_by_joint, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
