#!/usr/bin/env python3
"""Interactive SAM2 segmentation for grasp5 frame 0 at 540p.

VNC: connect to port 5912, display :12.
Controls:
  t/g: switch tissue/ground
  left click: positive point
  right click: negative point
  u: undo last point in current mode
  s: save current mode mask
  c: clear current mode points
  r: reset all points and saved masks
  q: quit and write saved masks/overlays
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("DISPLAY", ":12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

REPO = Path(__file__).resolve().parents[1]
SAM2_REPO = Path("/home/hsieh/data0/wad/embodied_gaussians_fixed_super2/third_party/sam2")
CHECKPOINT = Path("/home/hsieh/data0/wad/embodied_gaussians_fixed_super2/checkpoints/sam2/sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMAGE_PATH = REPO / "data/super/grasp5_native/rgb/000000-left-540.png"
DEPTH_PATH = Path("/home/hsieh/data0/wad/depth/000000.npy")
MASK_DIR = REPO / "data/super/grasp5_native/masks"
OVERLAY_DIR = MASK_DIR / "overlays"
PROMPT_PATH = MASK_DIR / "000000-sam2-clicks.json"

sys.path.insert(0, str(SAM2_REPO.resolve()))
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise SystemExit(f"Failed to read image: {IMAGE_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
if DEPTH_PATH.exists():
    depth = np.load(DEPTH_PATH)
    print(f"Depth: {DEPTH_PATH} shape={depth.shape} min={np.nanmin(depth):.6f} median={np.nanmedian(depth):.6f} max={np.nanmax(depth):.6f}")
    if depth.shape != (h, w):
        print(f"WARNING: depth shape {depth.shape} != image shape {(h, w)}")
else:
    print(f"WARNING: depth missing: {DEPTH_PATH}")

print(f"Loading SAM2 on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}...")
model = build_sam2(MODEL_CFG, str(CHECKPOINT), device="cuda")
predictor = SAM2ImagePredictor(model)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img_rgb)
print("SAM2 ready.")

mode = "tissue"
modes = {
    "tissue": {"points": [], "labels": [], "mask": None, "color": np.array([255, 40, 40], dtype=np.uint8)},
    "ground": {"points": [], "labels": [], "mask": None, "color": np.array([40, 255, 70], dtype=np.uint8)},
}
saved = {"tissue": None, "ground": None}


def choose_mask(masks, scores):
    return masks[int(np.argmax(np.asarray(scores).reshape(-1)))].astype(bool)


def predict_mask(name: str) -> None:
    data = modes[name]
    if not data["points"]:
        data["mask"] = None
        return
    pts = np.asarray(data["points"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True,
        )
    data["mask"] = choose_mask(masks, scores)
    print(f"{name}: points={len(pts)} mask_px={int(data['mask'].sum())} score={float(np.max(scores)):.4f}", flush=True)


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray | None, color: np.ndarray, alpha: float) -> np.ndarray:
    out = base_rgb.copy()
    if mask is not None:
        out[mask] = ((1.0 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def draw() -> None:
    canvas = img_rgb.copy()
    for name in ["tissue", "ground"]:
        if saved[name] is not None:
            canvas = make_overlay(canvas, saved[name], modes[name]["color"], 0.32)
    if modes[mode]["mask"] is not None:
        canvas = make_overlay(canvas, modes[mode]["mask"], modes[mode]["color"], 0.55)

    for name, data in modes.items():
        for (x, y), label in zip(data["points"], data["labels"]):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(canvas, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), 1, cv2.LINE_AA)

    lines = [
        f"MODE: {mode.upper()} | left=positive right=negative | t/g switch | u undo | s save | c clear | r reset | q quit",
        f"tissue pts={len(modes['tissue']['points'])} saved={saved['tissue'] is not None} | ground pts={len(modes['ground']['points'])} saved={saved['ground'] is not None}",
    ]
    y0 = h - 48
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("grasp5 SAM2 frame0 540p", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def on_mouse(event, x, y, flags, userdata) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        modes[mode]["points"].append([int(x), int(y)])
        modes[mode]["labels"].append(1)
        predict_mask(mode)
        draw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        modes[mode]["points"].append([int(x), int(y)])
        modes[mode]["labels"].append(0)
        predict_mask(mode)
        draw()


def write_outputs() -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "image": str(IMAGE_PATH),
        "depth": str(DEPTH_PATH),
        "shape": [h, w],
        "modes": {},
    }
    for name in ["tissue", "ground"]:
        data = modes[name]
        payload["modes"][name] = {"points": data["points"], "labels": data["labels"], "saved": saved[name] is not None}
        if saved[name] is None:
            print(f"{name}: NOT SAVED")
            continue
        mask = saved[name].astype(np.uint8) * 255
        mask_path = MASK_DIR / f"000000-{name}.png"
        overlay_path = OVERLAY_DIR / f"000000-{name}.png"
        cv2.imwrite(str(mask_path), mask)
        overlay = make_overlay(img_rgb, saved[name], data["color"], 0.45)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"{name}: {int(saved[name].sum())} px -> {mask_path}")
        print(f"overlay -> {overlay_path}")
    with open(PROMPT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"clicks -> {PROMPT_PATH}")


cv2.namedWindow("grasp5 SAM2 frame0 540p", cv2.WINDOW_NORMAL)
cv2.resizeWindow("grasp5 SAM2 frame0 540p", 1200, 700)
cv2.setMouseCallback("grasp5 SAM2 frame0 540p", on_mouse)
draw()

print("\nVNC ready: connect to port 5912 (display :12).")
print("Use t/g to switch. Save each mask with s. Quit with q.\n", flush=True)

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == 255:
        continue
    ch = chr(key).lower() if key < 128 else ""
    if ch == "t":
        mode = "tissue"
        print("mode=tissue")
    elif ch == "g":
        mode = "ground"
        print("mode=ground")
    elif ch == "u":
        if modes[mode]["points"]:
            modes[mode]["points"].pop()
            modes[mode]["labels"].pop()
            predict_mask(mode)
    elif ch == "s":
        if modes[mode]["mask"] is not None:
            saved[mode] = modes[mode]["mask"].copy()
            print(f"saved {mode}: {int(saved[mode].sum())} px")
        else:
            print(f"no current mask for {mode}")
    elif ch == "c":
        modes[mode]["points"].clear()
        modes[mode]["labels"].clear()
        modes[mode]["mask"] = None
        print(f"cleared {mode}")
    elif ch == "r":
        for name in modes:
            modes[name]["points"].clear()
            modes[name]["labels"].clear()
            modes[name]["mask"] = None
            saved[name] = None
        print("reset all")
    elif ch == "q":
        break
    draw()

cv2.destroyAllWindows()
write_outputs()
print("Done.")
