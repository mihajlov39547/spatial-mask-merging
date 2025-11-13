"""
evaluation.py — SMM Evaluation Script
====================================

Vectorized evaluator for instance-mask predictions vs. ground truth.
- Supports GPU acceleration with PyTorch (optional).
- Computes Precision, Recall, F1, Dice, PQ, Avg Fragments, Count Error, and Mean Error.
- Robust polygon->mask conversion with optional downscaling for speed.

Usage
-----
python evaluation.py \
  --pred_dir <predictions_dir> \
  --gt_dir <ground_truth_dir> \
  --img_dir <images_dir> \
  --out_csv ./results.csv \
  --iou_thr 0.5 \
  --downscale 4

Input Format
------------
Assumes prediction/GT JSON files share basenames. Each JSON contains:
{
  "image_name": "xxx.png",
  "image_size": [H, W],                # optional (fallback to --img_dir/filename)
  "annotations": [
    {
      "type": "car",                   # or "label"
      "class_id": 1,                   # optional; inferred from "type" otherwise
      "segmentation": [[x,y],...],     # or list of polygons [[...],[...]]
      "confidence": 0.93               # optional (predictions)
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import shared GPU evaluation utilities
from gpu_evaluation import (
    compute_metrics_gpu as compute_metrics_and_mean_error_torch,
    compute_metrics_cpu as compute_metrics_for_image_cpu,
    USE_CUDA as USE_TORCH,
    DEVICE
)

# Optional: Import torch for OOM handling
try:
    import torch
except Exception:
    torch = None

# Print GPU status
if USE_TORCH:
    print(f"Using CUDA for Evaluation: {USE_TORCH} (Device: {DEVICE})")


# =====================
# Helpers
# =====================
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_image_shape(img_dir: str, image_name: str, fallback_size: Tuple[int, int] = None) -> Tuple[int, int]:
    """Get image shape from JSON metadata or by reading the image file."""
    if fallback_size and all(isinstance(x, int) for x in fallback_size):
        return tuple(fallback_size)
    img_path = os.path.join(img_dir, image_name)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            return (h, w)
    # Final fallback - warn user
    print(f"⚠️  Warning: Could not determine size for {image_name}, using default (1024, 1024)")
    return (1024, 1024)


# =====================
# Batch Evaluation
# =====================
def evaluate_dir(
    pred_dir: str,
    gt_dir: str,
    img_dir: str,
    out_csv: str,
    iou_thr: float = 0.5,
    downscale: int = 1,
) -> None:
    """Evaluate predictions against ground truth for all images in directory."""
    # Validate inputs
    if not os.path.isdir(pred_dir):
        raise ValueError(f"Prediction directory not found: {pred_dir}")
    if not os.path.isdir(gt_dir):
        raise ValueError(f"Ground truth directory not found: {gt_dir}")
    if not os.path.isdir(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json")])
    
    if not pred_files:
        raise ValueError(f"No JSON files found in {pred_dir}")
    
    header_written = os.path.exists(out_csv)

    for fname in tqdm(pred_files, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"⚠️  Warning: No GT file for {fname}, skipping")
            continue

        try:
            pred = read_json(pred_path)
            gt   = read_json(gt_path)
        except json.JSONDecodeError as e:
            print(f"❌ Error reading {fname}: {e}")
            continue

        image_size = tuple(pred.get("image_size", [])) or None
        image_name = pred.get("image_name", os.path.splitext(fname)[0] + ".png")
        H, W = get_image_shape(img_dir, image_name, image_size)

        try:
            if USE_TORCH:
                metrics = compute_metrics_and_mean_error_torch(gt.get("annotations", []),
                                                               pred.get("annotations", []),
                                                               (H, W),
                                                               iou_thr=iou_thr,
                                                               downscale=downscale)
            else:
                metrics = compute_metrics_for_image_cpu(gt.get("annotations", []),
                                                        pred.get("annotations", []),
                                                        (H, W),
                                                        iou_thr=iou_thr,
                                                        downscale=downscale)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  GPU OOM for {fname}, trying CPU fallback...")
                if torch is not None:
                    torch.cuda.empty_cache()
                metrics = compute_metrics_for_image_cpu(gt.get("annotations", []),
                                                        pred.get("annotations", []),
                                                        (H, W),
                                                        iou_thr=iou_thr,
                                                        downscale=downscale)
            else:
                raise

        row = {
            "Image": fname,
            "Exec Time": pred.get("exec_time", 0),
            "Peak Memory (KB)": round(pred.get("peak_memory", 0), 2),
            "Predicted Objects": len(pred.get("annotations", [])),
            "GT Objects": len(gt.get("annotations", [])),
            **metrics
        }
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True

    print(f"\n✅ Metrics saved to: {out_csv}")


# =====================
# CLI
# =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="Directory with prediction JSONs")
    p.add_argument("--gt_dir", required=True, help="Directory with GT JSONs")
    p.add_argument("--img_dir", required=True, help="Directory with raw images for shape lookup")
    p.add_argument("--out_csv", required=True, help="Path to output CSV")
    p.add_argument("--iou_thr", type=float, default=0.5, help="Mask IoU threshold for TP matching")
    p.add_argument("--downscale", type=int, default=1, help="Downscale factor for masks (speed/VRAM)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Using CUDA for Evaluation: {USE_TORCH}")
    evaluate_dir(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        out_csv=args.out_csv,
        iou_thr=args.iou_thr,
        downscale=args.downscale,
    )


if __name__ == "__main__":
    main()
