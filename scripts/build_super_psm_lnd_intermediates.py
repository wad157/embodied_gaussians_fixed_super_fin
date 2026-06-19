#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TUNED_TRANSLATION_M = np.array(
    [-1.0187691605915499, -0.1690466995991076, -1.044475490077111], dtype=np.float64
) / 1000.0
DEFAULT_TUNED_RPY_RAD = np.radians(
    [-0.3519714768682627, 0.32665282898813774, 0.404393393425207]
).astype(np.float64)
JOINT_NAMES = [
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
]
D_BLENDER_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class ProjectionCandidate:
    name: str
    keypoints_field: str
    K_field: str = "K_left_rect"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fin-local SUPER PSM/LND model and motion intermediates.")
    ap.add_argument("--source", type=Path, default=REPO / "data/super/grasp5")
    ap.add_argument("--native", type=Path, default=REPO / "data/super/grasp5_native")
    ap.add_argument("--dataset", type=Path, default=REPO / "data/super/grasp5_offline_demo")
    ap.add_argument("--out", type=Path, default=REPO / "data/super/grasp5_offline_demo/instruments")
    ap.add_argument("--debug-frames", type=int, nargs="+", default=[0, 300, 800, 1200])
    ap.add_argument("--no-tuned-delta", action="store_true")
    return ap.parse_args()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_lnd_json_with_comments(path: Path) -> dict:
    text = path.read_text()
    text = re.sub(r"//.*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def parse_opencv_yaml_vectors(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text().replace("%YAML:1.0", "").replace("---", "")
    out = {}
    for match in re.finditer(r"(\w+):\s*\[(.*?)\]", text, flags=re.DOTALL):
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", match.group(2))]
        out[match.group(1)] = np.asarray(values, dtype=np.float64)
    return out


def rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float64)


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)


def transx(x: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64); t[0, 3] = x; return t


def transz(z: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64); t[2, 3] = z; return t


def modified_dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    return rotx(alpha) @ transx(a) @ rotz(theta) @ transz(d)


def build_fk(lnd: dict, q: np.ndarray) -> dict[int, np.ndarray]:
    transforms = {0: np.eye(4, dtype=np.float64)}
    t = np.eye(4, dtype=np.float64)
    for i, dh in enumerate(lnd["DH_params"], start=1):
        theta = float(dh.get("theta", 0.0))
        d = float(dh.get("D", 0.0))
        offset = float(dh.get("offset", 0.0))
        if dh["type"] == "revolute":
            theta += float(q[i - 1]) + offset
        elif dh["type"] == "prismatic":
            d += float(q[i - 1]) + offset
        else:
            raise ValueError(f"unsupported DH joint type: {dh['type']}")
        t = t @ modified_dh(float(dh.get("alpha", 0.0)), float(dh.get("A", 0.0)), theta, d)
        transforms[i] = t.copy()
    jaw = float(np.clip(q[6], -1.2, 1.2)) if len(q) > 6 else 0.0
    transforms[7] = transforms[6] @ rotz(0.5 * jaw)
    transforms[8] = transforms[6] @ rotz(-0.5 * jaw)
    return transforms


def transform_point(T: np.ndarray, xyz) -> np.ndarray:
    return (T @ np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64))[:3]


def lnd_points_and_lines(lnd: dict, link_t: dict[int, np.ndarray]) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    points = {}
    lines = []
    for item in lnd.get("point_features", []):
        points[item["name"]] = transform_point(link_t[int(item["link"])], item["position"])
    for i, item in enumerate(lnd.get("skeleton_structure", [])):
        a, b = f"skeleton_{i}_a", f"skeleton_{i}_b"
        points[a] = transform_point(link_t[int(item["link1"])], item["position1"])
        points[b] = transform_point(link_t[int(item["link2"])], item["position2"])
        lines.append((a, b))
    for item in lnd.get("shaft_features", []):
        link = int(item["link"])
        origin = transform_point(link_t[link], item["position"])
        direction = np.asarray(item["direction"], dtype=np.float64)
        direction = link_t[link][:3, :3] @ direction
        direction /= np.linalg.norm(direction) + 1e-12
        a, b = f"{item['name']}_near", f"{item['name']}_far"
        points[a] = origin - 0.08 * direction
        points[b] = origin + 0.08 * direction
        lines.append((a, b))
    return points, lines


def rodrigues_from_rpy(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)


def make_delta_transform(use_tuned: bool) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if use_tuned:
        T[:3, :3] = rodrigues_from_rpy(*DEFAULT_TUNED_RPY_RAD)
        T[:3, 3] = DEFAULT_TUNED_TRANSLATION_M
    return T


def make_handeye_matrix(handeye: dict[str, np.ndarray], scale: float = 0.001) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(np.asarray(handeye["PSM1_rvec"], dtype=np.float64).reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rmat
    T[:3, 3] = np.asarray(handeye["PSM1_tvec"], dtype=np.float64).reshape(3) * scale
    return T


def to_list(x) -> list:
    return np.asarray(x, dtype=np.float64).tolist()


def transform_point_dict(T: np.ndarray, points: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: to_list((T @ np.r_[v, 1.0])[:3]) for k, v in points.items()}


def transform_link_dict(T: np.ndarray, links: dict[int, np.ndarray]) -> dict[str, list]:
    return {str(k): to_list(T @ v) for k, v in links.items()}


def load_or_recompute_rectified_calib(native: Path, source: Path) -> dict:
    calib_path = native / "calib_rectified.json"
    calib = load_json(calib_path)
    needs_update = any(k not in calib for k in ["R1", "R2", "Q"])
    if needs_update:
        fs = cv2.FileStorage(str(source / "camera_calibration.yaml"), cv2.FILE_STORAGE_READ)
        K1 = fs.getNode("K1").mat(); D1 = fs.getNode("D1").mat().flatten()
        K2 = fs.getNode("K2").mat(); D2 = fs.getNode("D2").mat().flatten()
        R = fs.getNode("R").mat()
        t_node = fs.getNode("T")
        T_vec = np.array([t_node.at(i).real() for i in range(t_node.size())], dtype=np.float64)
        fs.release()
        image_size = (1920, 1080)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T_vec, alpha=0)
        calib["R1"] = R1.tolist(); calib["R2"] = R2.tolist(); calib["Q"] = Q.tolist()
        calib["P1"] = P1.tolist(); calib["P2"] = P2.tolist()
        calib["K_rectified"] = P1[:3, :3].tolist()
        calib.setdefault("K_left_rect", (P1[:3, :3] / 2.0).tolist())
        calib.setdefault("original_size", [1920, 1080])
        calib.setdefault("rectified_size", [1920, 1080])
        calib["calibration_augmented_at"] = "2026-06-18"
        calib["calibration_augmented_reason"] = "Add R1/R2/Q required for PSM raw-to-rectified reconstruction."
        save_json(calib_path, calib)
    return calib


def robot_state_at(robots: dict, frame_time: float) -> dict:
    robot = robots["sheep"]
    ts = np.asarray(robot["states_timestamps"], dtype=np.float64)
    idx = int(np.argmin(np.abs(ts - frame_time)))
    state = robot["states"][idx]
    return {
        "index": idx,
        "timestamp": float(ts[idx]),
        "delta_t": float(ts[idx] - frame_time),
        "q": np.asarray(state["q"], dtype=np.float64),
        "dq": state.get("dq", []),
        "effort": state.get("effort", []),
        "joint_names": state.get("joint_names", JOINT_NAMES),
        "header_time": state.get("header_time"),
        "bag_time": state.get("bag_time"),
    }


def project_summary(points: dict[str, list[float]], K: np.ndarray, width: int, height: int) -> dict:
    pts = {k: np.asarray(v, dtype=np.float64) for k, v in points.items()}
    pixels = {}
    in_front = 0
    in_image = 0
    finite = 0
    for name, p in pts.items():
        if p[2] > 1e-9:
            in_front += 1
            uvw = K @ p
            uv = uvw[:2] / uvw[2]
            pixels[name] = uv
            if np.isfinite(uv).all():
                finite += 1
                if 0 <= uv[0] < width and 0 <= uv[1] < height:
                    in_image += 1
        else:
            pixels[name] = np.array([np.nan, np.nan], dtype=np.float64)
    uv_arr = np.array([v for v in pixels.values() if np.isfinite(v).all()], dtype=np.float64)
    z = np.array([p[2] for p in pts.values()], dtype=np.float64)
    return {
        "num_points": len(pts),
        "in_front": int(in_front),
        "finite_pixels": int(finite),
        "in_image": int(in_image),
        "z_range_m": [float(z.min()), float(z.max())],
        "z_mean_m": float(z.mean()),
        "uv_bbox": None if len(uv_arr) == 0 else [float(uv_arr[:,0].min()), float(uv_arr[:,1].min()), float(uv_arr[:,0].max()), float(uv_arr[:,1].max())],
        "uv_median": None if len(uv_arr) == 0 else [float(np.median(uv_arr[:,0])), float(np.median(uv_arr[:,1]))],
    }


def draw_overlay(image: np.ndarray, points: dict[str, list[float]], lines, K: np.ndarray, title: str, out: Path) -> None:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    pixels = {}
    for name, xyz in points.items():
        p = np.asarray(xyz, dtype=np.float64)
        if p[2] > 1e-9:
            uvw = K @ p
            pixels[name] = uvw[:2] / uvw[2]
        else:
            pixels[name] = np.array([np.nan, np.nan])
    for a, b in lines:
        pa, pb = pixels.get(a), pixels.get(b)
        if pa is None or pb is None or not np.isfinite(pa).all() or not np.isfinite(pb).all():
            continue
        cv2.line(overlay, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    for name, uv in pixels.items():
        if not np.isfinite(uv).all():
            continue
        u, v = np.round(uv).astype(int)
        color = (0, 80, 255) if 0 <= u < w and 0 <= v < h else (80, 80, 255)
        cv2.circle(overlay, (u, v), 4, color, -1, cv2.LINE_AA)
        if 0 <= u < w and 0 <= v < h:
            cv2.putText(overlay, name, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(overlay, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,255,40), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)


def camera_opencv_pose(camera: dict) -> np.ndarray:
    X_WC = np.asarray(camera["X_WC"], dtype=np.float64)
    # Current SUPER-fin cameras are written directly as OpenCV camera ->
    # right-handed table-world transforms. Older datasets stored Blender-style
    # camera poses and needed D_BLENDER_OPENCV. Detect the new convention by the
    # explicit coordinate_frame marker added during manual table realignment.
    if camera.get("coordinate_frame") == "right_handed_table_world_z_up_m":
        return X_WC
    return X_WC @ D_BLENDER_OPENCV


def table_points_to_camera(points: dict[str, list[float]], X_WC_opencv: np.ndarray) -> dict[str, list[float]]:
    X_CW = np.linalg.inv(X_WC_opencv)
    return {k: to_list((X_CW @ np.r_[v, 1.0])[:3]) for k, v in points.items()}


def build_state(frame: dict, state_match: dict, lnd: dict, transforms: dict[str, np.ndarray]) -> tuple[dict, list[tuple[str, str]]]:
    q = state_match["q"]
    link_base = build_fk(lnd, q)
    points_base, lines = lnd_points_and_lines(lnd, link_base)
    fields = {
        "keypoints_psm_base": {k: to_list(v) for k, v in points_base.items()},
        "keypoints_raw_camera": transform_point_dict(transforms["raw_from_base"], points_base),
        "keypoints_raw_camera_tuned": transform_point_dict(transforms["raw_tuned_from_base"], points_base),
        "keypoints_rectified_from_raw_camera": transform_point_dict(transforms["rect_from_base"], points_base),
        "keypoints_rectified_from_raw_camera_tuned": transform_point_dict(transforms["rect_from_tuned_base"], points_base),
        "keypoints_projection_space": transform_point_dict(transforms["raw_tuned_from_base"], points_base),
        "keypoints_table_from_rectified_raw": transform_point_dict(transforms["table_from_rect_base"], points_base),
        "keypoints_table_from_projection_space": transform_point_dict(transforms["table_from_projection_base"], points_base),
    }
    link_fields = {
        "link_transforms_psm_base": {str(k): to_list(v) for k, v in link_base.items()},
        "link_transforms_raw_camera": transform_link_dict(transforms["raw_from_base"], link_base),
        "link_transforms_raw_camera_tuned": transform_link_dict(transforms["raw_tuned_from_base"], link_base),
        "link_transforms_rectified_from_raw_camera": transform_link_dict(transforms["rect_from_base"], link_base),
        "link_transforms_rectified_from_raw_camera_tuned": transform_link_dict(transforms["rect_from_tuned_base"], link_base),
        "link_transforms_table_from_rectified_raw": transform_link_dict(transforms["table_from_rect_base"], link_base),
        "link_transforms_table_from_projection_space": transform_link_dict(transforms["table_from_projection_base"], link_base),
    }
    out = {
        "frame_index": int(frame["index"]),
        "timestamp": float(frame["timestamp"]),
        "left_path": frame["left_path"],
        "right_path": frame.get("right_path"),
        "joint_state_index": int(state_match["index"]),
        "joint_state_timestamp": float(state_match["timestamp"]),
        "joint_delta_t": float(state_match["delta_t"]),
        "q": q.tolist(),
        "dq": state_match["dq"],
        "effort": state_match["effort"],
        **fields,
        **link_fields,
    }
    return out, lines


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lnd = parse_lnd_json_with_comments(args.source / "LND.json")
    handeye = parse_opencv_yaml_vectors(args.source / "handeye.yaml")
    calib = load_or_recompute_rectified_calib(args.native, args.source)
    robots = load_json(args.dataset / "robots.json")
    timestamps = load_json(args.native / "timestamps.json")
    meta = load_json(args.dataset / "super_ground_z_axis_realign.json")
    cameras = load_json(args.dataset / "cameras.json")

    K_full = np.asarray(calib.get("K_rectified", calib["P1"])[:3], dtype=np.float64)
    K_half = np.asarray(calib.get("K_left_rect", K_full), dtype=np.float64)
    # The current RGB files and videos are 1920x1080, so use full-size K for overlays.
    K_overlay = np.asarray(load_json(args.dataset / "videos/stereo_left.json")["K"], dtype=np.float64)
    R1 = np.asarray(calib["R1"], dtype=np.float64)
    T_rect_raw = np.eye(4, dtype=np.float64); T_rect_raw[:3, :3] = R1
    T_raw_base = make_handeye_matrix(handeye, 0.001)
    T_raw_tuned_base = make_delta_transform(not args.no_tuned_delta) @ T_raw_base
    T_rect_base = T_rect_raw @ T_raw_base
    T_rect_tuned_base = T_rect_raw @ T_raw_tuned_base
    if "coordinate_fix" in meta and "X_right_handed_table_from_rectified_camera_opencv" in meta["coordinate_fix"]:
        T_table_rect = np.asarray(meta["coordinate_fix"]["X_right_handed_table_from_rectified_camera_opencv"], dtype=np.float64)
    else:
        T_table_rect = np.asarray(meta["X_right_handed_table_from_rectified_camera_opencv"], dtype=np.float64)
    # For projection_space we intentionally preserve the raw_camera_tuned image-plane convention used by SUPER tuning.
    T_table_projection = T_table_rect.copy()
    transforms = {
        "raw_from_base": T_raw_base,
        "raw_tuned_from_base": T_raw_tuned_base,
        "rect_from_base": T_rect_base,
        "rect_from_tuned_base": T_rect_tuned_base,
        "table_from_rect_base": T_table_rect @ T_rect_base,
        "table_from_projection_base": T_table_projection @ T_raw_tuned_base,
    }

    states = []
    lines = []
    for frame in timestamps["frames"]:
        match = robot_state_at(robots, float(frame["timestamp"]))
        state, lines = build_state(frame, match, lnd, transforms)
        states.append(state)

    model = {
        "name": "psm1_lnd",
        "type": "fin_local_super_lnd_psm_kinematic_model",
        "generated_at": "2026-06-19",
        "source_files": {
            "lnd": str(args.source / "LND.json"),
            "handeye": str(args.source / "handeye.yaml"),
            "camera_calibration": str(args.source / "camera_calibration.yaml"),
            "native_calibration": str(args.native / "calib_rectified.json"),
            "robots": str(args.dataset / "robots.json"),
        },
        "joint_names": JOINT_NAMES,
        "dh_convention": "Craig modified DH: Rx(alpha) Tx(A) Rz(theta) Tz(D)",
        "jaw_model": "link7/link8 attached to link6 with +/-0.5*jaw z-rotation, matching psm_mimic_map.json intent",
        "default_projection_space": "raw_camera_tuned",
        "spaces_warning": "raw_camera_tuned/projection_space is currently the only historically visible PSM image space; strict rectified/table alignment remains under diagnosis.",
        "dh_params": lnd.get("DH_params", []),
        "point_features": lnd.get("point_features", []),
        "skeleton_structure": lnd.get("skeleton_structure", []),
        "shaft_features": lnd.get("shaft_features", []),
        "K_rectified_full": to_list(K_full),
        "K_left_rect_half": to_list(K_half),
        "R1_raw_to_rectified_left": to_list(R1),
        "T_raw_camera_psm_base": to_list(T_raw_base),
        "T_raw_camera_tuned_psm_base": to_list(T_raw_tuned_base),
        "T_rectified_from_raw_camera": to_list(T_rect_raw),
        "T_rectified_camera_psm_base": to_list(T_rect_base),
        "T_rectified_camera_tuned_psm_base": to_list(T_rect_tuned_base),
        "T_right_handed_table_from_rectified_camera": to_list(T_table_rect),
        "T_right_handed_table_from_rectified_psm_base": to_list(T_table_rect @ T_rect_base),
        "T_right_handed_table_from_projection_psm_base": to_list(T_table_projection @ T_raw_tuned_base),
    }
    motion = {
        "name": "psm1_lnd_motion",
        "type": "fin_local_super_lnd_psm_motion",
        "model_path": "psm1_lnd_model.json",
        "projection_space": "raw_camera_tuned",
        "num_frames": len(states),
        "num_joint_states_source": len(robots["sheep"]["states"]),
        "joint_names": JOINT_NAMES,
        "skeleton_line_names": [[a, b] for a, b in lines],
        "states": states,
    }
    save_json(args.out / "psm1_lnd_model.json", model)
    save_json(args.out / "psm1_lnd_motion.json", motion)

    candidates = [
        ProjectionCandidate("raw_camera", "keypoints_raw_camera"),
        ProjectionCandidate("raw_camera_tuned", "keypoints_raw_camera_tuned"),
        ProjectionCandidate("projection_space", "keypoints_projection_space"),
        ProjectionCandidate("rectified_from_raw_camera", "keypoints_rectified_from_raw_camera"),
        ProjectionCandidate("rectified_from_raw_camera_tuned", "keypoints_rectified_from_raw_camera_tuned"),
    ]
    debug_dir = args.out / "psm1_lnd_projection_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": "2026-06-19",
        "calib_rectified_augmented": all(k in calib for k in ["R1", "R2", "Q"]),
        "num_frames": len(states),
        "num_joint_states_source": len(robots["sheep"]["states"]),
        "debug_frames": args.debug_frames,
        "candidate_summaries": {},
        "note": "Counts are fin-local and no longer read embodied_gaussians_fixed_super2.",
    }
    for frame_index in args.debug_frames:
        if frame_index < 0 or frame_index >= len(states):
            continue
        state = states[frame_index]
        image = cv2.imread(str(args.native / state["left_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame_summary = {}
        for cand in candidates:
            points = state[cand.keypoints_field]
            summary = project_summary(points, K_overlay, image.shape[1], image.shape[0])
            frame_summary[cand.name] = summary
            draw_overlay(image, points, lines, K_overlay, f"{cand.name} frame {frame_index}", debug_dir / f"frame{frame_index:06d}_{cand.name}.png")
        # table points are projected by going table -> current left camera.
        X_WC = camera_opencv_pose(cameras["stereo_left"])
        for name, field in [
            ("table_from_rectified_raw_reprojected", "keypoints_table_from_rectified_raw"),
            ("table_from_projection_space_reprojected", "keypoints_table_from_projection_space"),
        ]:
            cam_points = table_points_to_camera(state[field], X_WC)
            frame_summary[name] = project_summary(cam_points, K_overlay, image.shape[1], image.shape[0])
        report["candidate_summaries"][str(frame_index)] = frame_summary
    save_json(args.out / "psm1_lnd_generation_report.json", report)
    print(json.dumps({
        "model": str(args.out / "psm1_lnd_model.json"),
        "motion": str(args.out / "psm1_lnd_motion.json"),
        "report": str(args.out / "psm1_lnd_generation_report.json"),
        "frames": len(states),
        "candidate_frame0": report["candidate_summaries"].get("0", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
