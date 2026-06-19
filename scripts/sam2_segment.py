#!/usr/bin/env python3
"""
SAM2 交互式点选分割 — 用鼠标点击来标记组织和地面。

操作:
  当前模式显示在窗口标题栏。
  按 t → 切换到 tissue 模式（左键点组织，右键点非组织）
  按 g → 切换到 ground 模式（左键点桌面，右键点非桌面）
  左键点击 → 正样本（这属于目标）
  右键点击 → 负样本（这不属于目标）
  按 s → 保存当前模式的 mask
  按 c → 清除当前模式的点击点
  按 r → 重新显示原始图像
  按 q → 退出并保存所有 mask

VNC: 服务器地址:5912
"""

import os, cv2, json, numpy as np
from pathlib import Path

os.environ.setdefault("DISPLAY", ":99")

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "data" / "super" / "grasp5_native"
RGB_DIR = NATIVE_DIR / "rgb"
MASK_DIR = NATIVE_DIR / "masks"
MASK_DIR.mkdir(exist_ok=True)

CHECKPOINT = "/home/hsieh/data0/wad/embodied_gaussians_fixed_super2/checkpoints/sam2/sam2.1_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

# ── 加载 SAM2 ──────────────────────────────────────────────
print("Loading SAM2...")
predictor = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, device="cuda"))
print("SAM2 loaded.")

# ── 加载图像 ───────────────────────────────────────────────
img_bgr = cv2.imread(str(RGB_DIR / "000000-left.png"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

predictor.set_image(img_rgb)
print(f"Image: {w}x{h}, ready.")

# ── 状态管理 ───────────────────────────────────────────────
mode = "tissue"  # "tissue" or "ground"
modes = {
    "tissue": {"points": [], "labels": [], "mask": None, "color": (0, 255, 0)},
    "ground": {"points": [], "labels": [], "mask": None, "color": (255, 0, 0)},
}
mask_history = {"tissue": None, "ground": None}

def predict_mask(mode_name):
    """用当前点击点预测 mask。"""
    data = modes[mode_name]
    if not data["points"]:
        data["mask"] = None
        return
    pts = np.array(data["points"], dtype=np.float32)
    lbls = np.array(data["labels"], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbls,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    data["mask"] = masks[best_idx].astype(np.uint8) * 255

def draw():
    """重绘窗口。"""
    canvas = img_rgb.copy()
    # 画已保存的 mask
    for mn in ["tissue", "ground"]:
        m = mask_history.get(mn)
        if m is not None:
            overlay = np.zeros_like(canvas)
            color = modes[mn]["color"]
            overlay[m > 0] = color
            canvas = cv2.addWeighted(canvas, 1, overlay, 0.35, 0)

    # 画当前模式的预览 mask
    cur = modes[mode]
    if cur["mask"] is not None:
        overlay = np.zeros_like(canvas)
        overlay[cur["mask"] > 0] = cur["color"]
        canvas = cv2.addWeighted(canvas, 1, overlay, 0.5, 0)

    # 画所有点击点
    for mn, data in modes.items():
        for (px, py), label in zip(data["points"], data["labels"]):
            c = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(canvas, (px, py), 8, c, -1)
            cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1)

    # 信息栏
    lines = [
        f"MODE: {mode.upper()} | 左键=正样本  右键=负样本 | t/g切换  s保存  c清除  q退出",
        f"tissue: {len(modes['tissue']['points'])} pts | ground: {len(modes['ground']['points'])} pts",
    ]
    if mask_history.get("tissue") is not None:
        lines.append(f"tissue mask: SAVED ({mask_history['tissue'].sum() // 255} px)")
    if mask_history.get("ground") is not None:
        lines.append(f"ground mask: SAVED ({mask_history['ground'].sum() // 255} px)")

    y0 = h - 20 - 25 * len(lines)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (15, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("SAM2 Segmentation", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        modes[mode]["points"].append([x, y])
        modes[mode]["labels"].append(1)  # positive
        predict_mask(mode)
    elif event == cv2.EVENT_RBUTTONDOWN:
        modes[mode]["points"].append([x, y])
        modes[mode]["labels"].append(0)  # negative
        predict_mask(mode)
    draw()

cv2.namedWindow("SAM2 Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SAM2 Segmentation", 1200, 700)
cv2.setMouseCallback("SAM2 Segmentation", mouse_callback)
draw()

print("\n窗口已打开，请通过 VNC :5912 操作。")
print("  t = tissue模式  g = ground模式")
print("  左键 = 正样本  右键 = 负样本")
print("  s = 保存当前mask  q = 退出\n")

while True:
    key = cv2.waitKey(50) & 0xFF
    ch = chr(key).lower() if key < 128 else ""

    if ch == 't':
        mode = "tissue"
    elif ch == 'g':
        mode = "ground"
    elif ch == 's':
        cur = modes[mode]
        if cur["mask"] is not None:
            mask_history[mode] = cur["mask"].copy()
            print(f"  Saved {mode} mask: {cur['mask'].sum()//255} px")
        else:
            print(f"  No mask to save for {mode}")
    elif ch == 'c':
        modes[mode]["points"] = []
        modes[mode]["labels"] = []
        modes[mode]["mask"] = None
    elif ch == 'r':
        # Reset everything
        for m in modes.values():
            m["points"] = []
            m["labels"] = []
            m["mask"] = None
        mask_history = {"tissue": None, "ground": None}
    elif ch == 'q':
        break

    draw()

cv2.destroyAllWindows()

# ── 保存 ──────────────────────────────────────────────────
for mn in ["tissue", "ground"]:
    m = mask_history.get(mn)
    if m is not None:
        path = MASK_DIR / f"000000-{mn}.png"
        cv2.imwrite(str(path), m)
        print(f"  {mn}: {m.sum()//255} px → {path}")
    else:
        print(f"  {mn}: NOT SAVED")

print("\nDone!")
