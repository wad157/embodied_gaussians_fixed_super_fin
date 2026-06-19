#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_super_psm_validation as psm


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    # Inverse of URDF's fixed-axis rpy convention as implemented by rpy_matrix:
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-R[0, 1]), float(R[1, 1]))
    return roll, pitch, yaw


def weighted_kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights.astype(np.float64)
    weights = weights / weights.sum()
    c_src = (src * weights[:, None]).sum(axis=0)
    c_dst = (dst * weights[:, None]).sum(axis=0)
    A = (src - c_src) * weights[:, None]
    B = dst - c_dst
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R
    X[:3, 3] = t
    return X


def residuals(X: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pred = (X[:3, :3] @ src.T + X[:3, 3:4]).T
    return np.linalg.norm(pred - dst, axis=1)


def set_fixed_origin_identity(children: dict) -> None:
    for joints in children.values():
        for joint in joints:
            if joint["name"] == "fixed":
                joint["origin"] = np.eye(4, dtype=np.float64)


def update_fixed_joint_origin(urdf_path: Path, xyz: np.ndarray, rpy: tuple[float, float, float]) -> None:
    text = urdf_path.read_text()
    xyz_s = " ".join(f"{v:.9g}" for v in xyz)
    rpy_s = " ".join(f"{v:.9g}" for v in rpy)
    pat = r'(<joint name="fixed" type="fixed">.*?<origin )rpy="[^"]*" xyz="[^"]*"(\s*/>)'
    repl = rf'\1rpy="{rpy_s}" xyz="{xyz_s}"\2'
    new_text, n = re.subn(pat, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("Could not find fixed joint origin in URDF")
    urdf_path.write_text(new_text)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    urdf_path = repo / "data/super/psm_robot/psm.urdf"
    robots_path = repo / "data/super/grasp5_offline_demo/robots.json"
    mimic_path = repo / "data/super/psm_robot/psm_mimic_map.json"
    motion_path = repo / "data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json"
    report_path = repo / "data/super/psm_robot/psm_base_correction_report.json"

    links, children, _ = psm.parse_urdf(urdf_path)
    set_fixed_origin_identity(children)

    robots = json.load(open(robots_path))
    q = robots["sheep"]["states"][0]["q"]
    mimic = json.load(open(mimic_path))
    q_by_joint = psm.expand_super_q_to_urdf_joints(q, mimic)
    tf_local = psm.compute_link_transforms(children, q_by_joint)

    state0 = json.load(open(motion_path))["states"][0]
    lnd_links = {
        name: np.asarray(T, dtype=np.float64)[:3, 3]
        for name, T in state0["link_transforms_table_from_projection_space"].items()
    }
    lnd_pts = {
        name: np.asarray(pt, dtype=np.float64)
        for name, pt in state0["keypoints_table_from_projection_space"].items()
    }

    # The distal LND link and ee keypoints are close but not identical. Averaging them
    # avoids overfitting to one noisy projected landmark.
    distal_target = np.vstack([
        lnd_links["6"],
        lnd_pts["ee_front"],
        lnd_pts["ee_back"],
        lnd_pts["skeleton_4_a"],
        lnd_pts["skeleton_4_b"],
    ]).mean(axis=0)

    anchors = [
        {
            "name": "base_origin_to_lnd_base",
            "src_link": "PSM1_outer_pitch_link",
            "src": tf_local["PSM1_outer_pitch_link"][:3, 3],
            "dst": lnd_links["0"],
            "weight": 0.7,
        },
        {
            "name": "tool_main_to_lnd_insertion_link3",
            "src_link": "PSM1_tool_main_link",
            "src": tf_local["PSM1_tool_main_link"][:3, 3],
            "dst": lnd_links["3"],
            "weight": 1.3,
        },
        {
            "name": "wrist_to_lnd_wrist_link4",
            "src_link": "PSM1_tool_wrist_link",
            "src": tf_local["PSM1_tool_wrist_link"][:3, 3],
            "dst": lnd_links["4"],
            "weight": 1.5,
        },
        {
            "name": "distal_shaft_to_lnd_distal",
            "src_link": "PSM1_tool_wrist_sca_shaft_link",
            "src": tf_local["PSM1_tool_wrist_sca_shaft_link"][:3, 3],
            "dst": distal_target,
            "weight": 1.5,
        },
        {
            "name": "tool_tip_to_lnd_distal",
            "src_link": "PSM1_tool_tip_link",
            "src": tf_local["PSM1_tool_tip_link"][:3, 3],
            "dst": distal_target,
            "weight": 0.8,
        },
    ]

    src = np.vstack([a["src"] for a in anchors])
    dst = np.vstack([a["dst"] for a in anchors])
    weights = np.asarray([a["weight"] for a in anchors], dtype=np.float64)

    X = weighted_kabsch(src, dst, weights)
    res = residuals(X, src, dst)
    rpy = matrix_to_rpy(X[:3, :3])
    update_fixed_joint_origin(urdf_path, X[:3, 3], rpy)

    for a, e in zip(anchors, res):
        pred = X[:3, :3] @ a["src"] + X[:3, 3]
        a["src"] = a["src"].tolist()
        a["dst"] = a["dst"].tolist()
        a["pred"] = pred.tolist()
        a["residual_m"] = float(e)

    report = {
        "method": "weighted_rigid_fit_from_urdf_fk_identity_base_to_lnd_table_world_frame0",
        "frame_index": int(state0["frame_index"]),
        "source_files": {
            "urdf": str(urdf_path.relative_to(repo)),
            "robots": str(robots_path.relative_to(repo)),
            "mimic_map": str(mimic_path.relative_to(repo)),
            "lnd_motion": str(motion_path.relative_to(repo)),
        },
        "new_fixed_joint_origin": {
            "xyz": X[:3, 3].tolist(),
            "rpy": list(rpy),
            "matrix": X.tolist(),
        },
        "anchors": anchors,
        "residual_summary_m": {
            "mean": float(res.mean()),
            "max": float(res.max()),
            "per_anchor": res.tolist(),
        },
        "notes": [
            "grip_far is not used because its z value is below the fitted table plane in frame0.",
            "This correction only changes the URDF fixed root pose; joint values and mimic joints are unchanged.",
            "The target frame is the fin-local right-handed table-world frame with z=0 table plane.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
