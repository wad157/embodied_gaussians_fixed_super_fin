#!/usr/bin/env python3
"""
Extract grasp5.bag → robots.json + rectified stereo PNG sequences.

Input:  data/super/grasp5/grasp5.bag + camera_calibration.yaml
Output: data/super/grasp5_native/
          ├── robots.json
          ├── calib_rectified.json
          ├── rgb/
          │   ├── 000000-left.png, 000000-right.png
          │   └── ...
          ├── timestamps_left.json
          └── timestamps_right.json
"""

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
GRASP5_DIR = REPO_ROOT / "data" / "super" / "grasp5"
BAG_PATH = GRASP5_DIR / "grasp5.bag"
CALIB_PATH = GRASP5_DIR / "camera_calibration.yaml"
OUT_DIR = REPO_ROOT / "data" / "super" / "grasp5_native"
RGB_DIR = OUT_DIR / "rgb"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RGB_DIR.mkdir(parents=True, exist_ok=True)

# ── load calibration ───────────────────────────────────────────────
# OpenCV YAML format uses %YAML:1.0 directive and !!opencv-matrix tags,
# which standard yaml can't parse. Use cv2.FileStorage instead.
fs = cv2.FileStorage(str(CALIB_PATH), cv2.FILE_STORAGE_READ)

K1 = fs.getNode("K1").mat()      # (3,3)
D1 = fs.getNode("D1").mat()      # (1,5) → flatten to (5,)
K2 = fs.getNode("K2").mat()      # (3,3)
D2 = fs.getNode("D2").mat()      # (1,5)
R  = fs.getNode("R").mat()       # (3,3)

# T is a plain list (not !!opencv-matrix), read it manually
t_node = fs.getNode("T")
T_vec = np.array([t_node.at(i).real() for i in range(t_node.size())], dtype=np.float64)

fs.release()

D1 = D1.flatten()
D2 = D2.flatten()

img_h, img_w = 1080, 1920  # from ImageSize: [1080, 1920]
print(f"Image size: {img_w}x{img_h}")
# T is in mm (OpenCV calibration convention for surgical cameras)
baseline_m = abs(T_vec[0]) / 1000.0  # mm → m
print(f"Stereo baseline: {baseline_m:.4f} m ({abs(T_vec[0]):.2f} mm)")

# ── stereo rectification ───────────────────────────────────────────
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, (img_w, img_h), R, T_vec,
    alpha=0,  # crop to only valid pixels (no black borders)
)
# Rectified intrinsics (3x3) from projection matrix P1
K_rectified = P1[:3, :3].copy()
print(f"K_rectified:\n{K_rectified}")

# Pre-compute remap tables (same for all frames)
map1_left_x, map1_left_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (img_w, img_h), cv2.CV_32FC1)
map1_right_x, map1_right_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (img_w, img_h), cv2.CV_32FC1)

# Get rectified image size (may be different from original if alpha=0)
rect_h, rect_w = map1_left_x.shape
print(f"Rectified image size: {rect_w}x{rect_h}")

# Save calibration
calib_rectified = {
    "K_rectified": K_rectified.tolist(),
    "P1": P1.tolist(),
    "P2": P2.tolist(),
    "R1": R1.tolist(),
    "R2": R2.tolist(),
    "baseline_m": float(baseline_m),
    "original_size": [img_w, img_h],
    "rectified_size": [rect_w, rect_h],
}
(OUT_DIR / "calib_rectified.json").write_text(json.dumps(calib_rectified, indent=2))
print("Saved calib_rectified.json")

# ── parse bag ──────────────────────────────────────────────────────
from rosbags.rosbag1 import Reader

joint_positions = []   # [(ts_sec, [q0..q6])]
joint_timestamps_ns = []

stereo_left_data = []  # [(ts_sec, ts_ns, raw_bytes)]
stereo_right_data = []

with Reader(BAG_PATH) as reader:
    for conn in reader.connections:
        print(f"Reading: {conn.topic} ({conn.msgcount} msgs)...")
        count = 0
        for conn_obj, ts_ns, raw_msg in reader.messages(connections=[conn]):
            if count == 0:
                first_ts = ts_ns
            ts_sec = (ts_ns - first_ts) / 1e9  # normalize from 0

            if "joint" in conn.topic:
                # Parse JointState manually
                offset = 0
                offset += 4  # seq
                stamp_secs = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                stamp_nsecs = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                frame_id_len = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                offset += frame_id_len  # skip frame_id

                num_names = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                for _ in range(num_names):
                    strlen = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                    offset += strlen

                num_pos = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
                positions = list(struct.unpack_from("<" + "d" * num_pos, raw_msg, offset))
                joint_positions.append((ts_sec, positions[:7]))
                joint_timestamps_ns.append(ts_ns)

            elif "slave/left" in conn.topic:
                stereo_left_data.append((ts_sec, ts_ns, raw_msg))

            elif "slave/right" in conn.topic:
                stereo_right_data.append((ts_sec, ts_ns, raw_msg))

            count += 1
        print(f"  → extracted {count} messages")

# ── write robots.json ──────────────────────────────────────────────
control = [pos for _, pos in joint_positions]
control_timestamps = [ts for ts, _ in joint_positions]
states = [{"q": pos} for _, pos in joint_positions]

robots_json = {
    "sheep": {
        "control": control,
        "control_timestamps": control_timestamps,
        "states": states,
        "states_timestamps": control_timestamps,
    }
}
(OUT_DIR / "robots.json").write_text(json.dumps(robots_json, indent=2))
print(f"Saved robots.json ({len(control)} timesteps)")

# ── write rectified stereo images ──────────────────────────────────
def parse_image(raw_msg):
    """Parse sensor_msgs/Image → (height, width, np.uint8 RGB array).

    Handles multiple encodings including cases where encoding string
    doesn't match the actual data layout.
    """
    offset = 0
    offset += 4  # seq
    offset += 4  # stamp secs
    offset += 4  # stamp nsecs
    frame_id_len = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    offset += frame_id_len

    height = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    width  = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    encoding_len = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    encoding = raw_msg[offset:offset+encoding_len].decode(); offset += encoding_len
    offset += 1  # is_bigendian
    step = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    data_len = struct.unpack_from("<I", raw_msg, offset)[0]; offset += 4
    img_data = raw_msg[offset:offset+data_len]

    # Reshape using actual step and height (not trusting width*channels)
    img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, step)

    bpp = step // width  # bytes per pixel
    if bpp == 3:
        # Standard RGB/BGR
        img_rgb = img[:, :width*3].reshape(height, width, 3)
        if "bgr" in encoding.lower():
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    elif bpp == 2:
        # YUV 4:2:2 or similar 2-byte-per-pixel format
        img_yuv = img[:, :width*2].reshape(height, width, 2)
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB_YUYV)
    elif bpp == 1:
        # Mono8 (grayscale)
        img_gray = img[:, :width].reshape(height, width)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unknown bpp={bpp} (step={step}, width={width})")

    return height, width, img_rgb

def rectify_and_save(raw_msg, map_x, map_y, out_path):
    """Parse, rectify, and save a single image."""
    h, w, img_rgb = parse_image(raw_msg)
    rectified = cv2.remap(img_rgb, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
    return rectified.shape

print(f"\nRectifying {len(stereo_left_data)} stereo pairs...")
timestamps_left = []
timestamps_right = []

num_pairs = min(len(stereo_left_data), len(stereo_right_data))
for i in range(num_pairs):
    ts_l, ts_ns_l, raw_l = stereo_left_data[i]
    ts_r, ts_ns_r, raw_r = stereo_right_data[i]

    out_left  = RGB_DIR / f"{i:06d}-left.png"
    out_right = RGB_DIR / f"{i:06d}-right.png"

    shape_l = rectify_and_save(raw_l, map1_left_x, map1_left_y, out_left)
    shape_r = rectify_and_save(raw_r, map1_right_x, map1_right_y, out_right)

    timestamps_left.append(ts_l)
    timestamps_right.append(ts_r)

    if i % 200 == 0:
        print(f"  frame {i}/{num_pairs} (ts={ts_l:.3f}s), rectified size: {shape_l[1]}x{shape_l[0]}")

print(f"Done. Saved {num_pairs} stereo pairs to {RGB_DIR}")

# ── save timestamp arrays ──────────────────────────────────────────
json.dump(timestamps_left, (OUT_DIR / "timestamps_left.json").open("w"), indent=2)
json.dump(timestamps_right, (OUT_DIR / "timestamps_right.json").open("w"), indent=2)

print(f"\nAll done. Output in {OUT_DIR}/")
print(f"  robots.json:              {OUT_DIR / 'robots.json'}")
print(f"  calib_rectified.json:     {OUT_DIR / 'calib_rectified.json'}")
print(f"  rgb/000000-left.png ...:  {num_pairs} left frames")
print(f"  rgb/000000-right.png ...: {num_pairs} right frames")
print(f"  timestamps_left.json:     {len(timestamps_left)} entries")
print(f"  timestamps_right.json:    {len(timestamps_right)} entries")
