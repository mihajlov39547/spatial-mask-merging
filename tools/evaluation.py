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

# Optional: PyTorch for GPU acceleration
try:
    import torch
    USE_TORCH = torch.cuda.is_available()
    DEVICE = torch.device("cuda") if USE_TORCH else torch.device("cpu")
except Exception:
    torch = None
    USE_TORCH = False
    DEVICE = None


# =====================
# Helpers
# =====================
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_image_shape(img_dir: str, image_name: str, fallback_size: Tuple[int, int] | None) -> Tuple[int, int]:
    if fallback_size and all(isinstance(x, int) for x in fallback_size):
        return tuple(fallback_size)
    img_path = os.path.join(img_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        # final fallback
        return (1024, 1024)
    h, w = img.shape[:2]
    return (h, w)


def polygons_to_mask(polygons: List[List[List[float]]], image_size: Tuple[int, int], downscale: int = 1) -> np.ndarray:
    """polygons: list of lists of [x,y]; image_size: (H,W). Returns bool mask."""
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts], 1)
    if downscale and downscale > 1:
        new_shape = (W // downscale, H // downscale)
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def load_mask_from_segmentation(segmentation, image_shape, downscale: int = 1) -> np.ndarray:
    if not segmentation or not isinstance(segmentation, list):
        return np.zeros(image_shape if downscale == 1 else (image_shape[0] // downscale, image_shape[1] // downscale), dtype=bool)
    polygons = segmentation if isinstance(segmentation[0][0], list) else [segmentation]
    return polygons_to_mask(polygons, image_shape, downscale=downscale)


def iou_masks(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum(dtype=np.int64)
    union = np.logical_or(m1, m2).sum(dtype=np.int64)
    return float(inter) / (float(union) + 1e-10)


def bbox_from_mask(mask: np.ndarray) -> List[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def ensure_class_ids(anns: List[Dict[str, Any]]) -> List[int]:
    """Return class_id list; infer from 'type'/'label' if missing, with stable mapping per call."""
    name_to_id: Dict[str, int] = {}
    ids: List[int] = []
    next_id = 0
    for a in anns:
        if "class_id" in a and isinstance(a["class_id"], (int, np.integer)):
            ids.append(int(a["class_id"]))
            continue
        name = str(a.get("type", a.get("label", "object")))
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1
        ids.append(name_to_id[name])
    return ids


# =====================
# GPU Evaluators
# =====================
def _pairwise_iou_masks_torch(gt_t, pr_t) -> "torch.Tensor":
    """gt_t:[G,H,W] bool, pr_t:[P,H,W] bool -> IoU [G,P] via matmul on flattened masks."""
    G, H, W = gt_t.shape
    P = pr_t.shape[0]
    if G == 0 or P == 0:
        return torch.zeros((G, P), dtype=torch.float32, device=gt_t.device)
    gt_f = gt_t.reshape(G, -1).to(torch.float32)
    pr_f = pr_t.reshape(P, -1).to(torch.float32)
    inter = gt_f @ pr_f.T
    area_g = gt_f.sum(dim=1)
    area_p = pr_f.sum(dim=1)
    union = area_g[:, None] + area_p[None, :] - inter
    return inter / (union + 1e-10)


def _boxes_from_stack_stable(mstk: "torch.Tensor") -> "torch.Tensor":
    """mstk: [N,H,W] bool -> boxes [N,4] float32 (xyxy)."""
    N, H, W = mstk.shape
    boxes = torch.zeros((N, 4), dtype=torch.float32, device=mstk.device)
    if N == 0:
        return boxes
    rows_any = mstk.any(dim=2)
    cols_any = mstk.any(dim=1)
    y_idx = torch.arange(H, device=mstk.device)
    x_idx = torch.arange(W, device=mstk.device)
    y_min = torch.where(rows_any, y_idx, H).amin(dim=1)
    y_max = torch.where(rows_any, y_idx, -1).amax(dim=1)
    x_min = torch.where(cols_any, x_idx, W).amin(dim=1)
    x_max = torch.where(cols_any, x_idx, -1).amax(dim=1)
    y_min = torch.clamp_min(y_min, 0); y_max = torch.clamp_min(y_max, 0)
    x_min = torch.clamp_min(x_min, 0); x_max = torch.clamp_min(x_max, 0)
    boxes[:, 0] = x_min.to(torch.float32)
    boxes[:, 1] = y_min.to(torch.float32)
    boxes[:, 2] = x_max.to(torch.float32)
    boxes[:, 3] = y_max.to(torch.float32)
    return boxes


def compute_metrics_and_mean_error_torch(
    gt_anns: List[Dict[str, Any]],
    pred_anns: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_thr: float = 0.5,
    downscale: int = 1
) -> Dict[str, float]:
    """GPU evaluator with chunking-free path (use if memory allows)."""
    # Build masks on CPU then move to DEVICE
    gt_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in gt_anns]
    pr_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in pred_anns]
    gt_cls   = ensure_class_ids(gt_anns)
    pr_cls   = ensure_class_ids(pred_anns)

    gt_by, pr_by = defaultdict(list), defaultdict(list)
    for i,c in enumerate(gt_cls): gt_by[c].append(i)
    for j,c in enumerate(pr_cls): pr_by[c].append(j)

    TP = 0
    total_iou = 0.0
    matched_gt, matched_pr = set(), set()
    fragments, mean_err_acc = [], []

    for c in set(gt_by) | set(pr_by):
        gi = gt_by.get(c, []); pj = pr_by.get(c, [])
        G, P = len(gi), len(pj)
        if G == 0:
            continue

        gt_np = np.stack([gt_masks[k] for k in gi], 0).astype(np.bool_)
        gt_t  = torch.from_numpy(gt_np).to(DEVICE)

        frag = torch.zeros(G, dtype=torch.int64, device=DEVICE)

        if P > 0:
            pr_np = np.stack([pr_masks[k] for k in pj], 0).astype(np.bool_)
            pr_t  = torch.from_numpy(pr_np).to(DEVICE)

            iou = _pairwise_iou_masks_torch(gt_t, pr_t)  # [G,P]
            vals, arg = torch.max(iou, dim=0)            # per pred: best GT
            keep = vals >= iou_thr

            if keep.any():
                sel_pr = torch.nonzero(keep, as_tuple=False).squeeze(1)
                sel_gt = arg[keep]
                frag += torch.bincount(sel_gt, minlength=G)
                TP += int(keep.sum().item())
                total_iou += float(vals[keep].sum().item())

                for jj in sel_pr.tolist(): matched_pr.add(pj[jj])
                for ii in sel_gt.unique().tolist(): matched_gt.add(gi[ii])

            # Mean Error via boxes
            gt_boxes = _boxes_from_stack_stable(gt_t)
            pr_boxes = _boxes_from_stack_stable(pr_t)
            gt_xy = gt_boxes[sel_gt, :2] if keep.any() else torch.empty((0,2), device=DEVICE)
            pr_xy = pr_boxes[sel_pr, :2] if keep.any() else torch.empty((0,2), device=DEVICE)
            if gt_xy.numel():
                err = torch.linalg.norm(gt_xy - pr_xy, dim=1)
                mean_err_acc.extend(err.detach().cpu().tolist())

            del pr_t, iou, vals, arg

        fragments.extend(frag.detach().cpu().tolist())
        del gt_t, frag

    FP = len(pr_masks) - len(matched_pr)
    FN = len(gt_masks) - len(matched_gt)
    precision = TP / (TP + FP + 1e-10)
    recall    = TP / (TP + FN + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    dice      = 2 * total_iou / (TP + 1e-10)
    avg_frag  = float(np.mean([f for f in fragments if f > 0])) if fragments else 0.0
    count_err = abs(len(pr_masks) - len(gt_masks))
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-10)
    sq = total_iou / (TP + 1e-10)
    pq = dq * sq
    mean_err = float(np.mean(mean_err_acc)) if mean_err_acc else 0.0

    return {
        "Precision": precision, "Recall": recall, "F1 Score": f1,
        "Dice Coefficient": dice, "Avg Fragments": avg_frag,
        "Count Error": count_err, "PQ": pq, "Mean Error": mean_err
    }


def compute_metrics_for_image_cpu(
    gt_anns: List[Dict[str, Any]],
    pred_anns: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_thr: float = 0.5,
    downscale: int = 1
) -> Dict[str, float]:
    gt_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in gt_anns]
    pr_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in pred_anns]
    gt_cls   = ensure_class_ids(gt_anns)
    pr_cls   = ensure_class_ids(pred_anns)

    matched_gt, matched_pr = set(), set()
    TP = 0
    total_iou = 0.0
    fragments = []

    for i, gm in enumerate(gt_masks):
        matched = 0
        for j, pm in enumerate(pr_masks):
            if pr_cls[j] != gt_cls[i] or j in matched_pr:
                continue
            iou = iou_masks(gm, pm)
            if iou >= iou_thr:
                matched += 1
                matched_gt.add(i)
                matched_pr.add(j)
                TP += 1
                total_iou += iou
        fragments.append(matched)

    FP = len(pr_masks) - len(matched_pr)
    FN = len(gt_masks) - len(matched_gt)
    precision = TP / (TP + FP + 1e-10)
    recall    = TP / (TP + FN + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    dice      = 2 * total_iou / (TP + 1e-10)
    avg_frag  = float(np.mean([f for f in fragments if f > 0])) if fragments else 0.0
    count_err = abs(len(pr_masks) - len(gt_masks))
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-10)
    sq = total_iou / (TP + 1e-10)
    pq = dq * sq

    return {
        "Precision": precision, "Recall": recall, "F1 Score": f1,
        "Dice Coefficient": dice, "Avg Fragments": avg_frag,
        "Count Error": count_err, "PQ": pq
    }


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
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json")])
    header_written = os.path.exists(out_csv)

    for fname in tqdm(pred_files, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

        pred = read_json(pred_path)
        gt   = read_json(gt_path)

        image_size = tuple(pred.get("image_size", [])) or None
        image_name = pred.get("image_name", os.path.splitext(fname)[0] + ".png")
        H, W = get_image_shape(img_dir, image_name, image_size)

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
