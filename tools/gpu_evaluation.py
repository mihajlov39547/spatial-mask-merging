"""
gpu_evaluation.py — Shared GPU-Accelerated Evaluation Functions
================================================================

Common GPU evaluation utilities for mask-based instance segmentation.
Used by both evaluation.py and optimize_smm.py.

Features:
- Vectorized mask IoU computation via GPU matmul
- Adaptive VRAM chunking to prevent OOM errors
- Pinned memory transfers for faster H2D
- Automatic CPU fallback when CUDA unavailable
- Box extraction from mask stacks on GPU
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# GPU acceleration
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
except Exception:
    torch = None
    USE_CUDA = False
    DEVICE = None

# GPU memory management settings
CHUNK_P = 1024   # pred-chunk size to control GPU memory
CHUNK_G = 256    # GT block size (tune per VRAM)
ADAPTIVE_P = True  # adaptive pred-chunk sizing from free VRAM
IOU_THRESHOLD = 0.5  # default IoU threshold for matching
DOWNSCALE_FACTOR = 4  # default downscale factor


# =====================
# Mask Processing
# =====================
def load_mask_from_segmentation(segmentation, image_shape: Tuple[int, int], downscale: int = 1) -> np.ndarray:
    """Convert polygon segmentation to binary mask with optional downscaling."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    if not segmentation or not isinstance(segmentation, list):
        if downscale > 1:
            return np.zeros((image_shape[0] // downscale, image_shape[1] // downscale), dtype=bool)
        return mask.astype(bool)

    polygons = segmentation if isinstance(segmentation[0][0], list) else [segmentation]
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 1)

    if downscale > 1:
        new_shape = (mask.shape[1] // downscale, mask.shape[0] // downscale)
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


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
# GPU Helper Functions
# =====================
def _pairwise_iou_masks_torch(gt_t, pr_t):
    """
    Compute IoU matrix between GT and pred masks using GPU matmul.
    gt_t: [G,H,W] bool, pr_t: [P,H,W] bool -> IoU [G,P]
    """
    G, H, W = gt_t.shape
    P = pr_t.shape[0]
    if G == 0 or P == 0:
        return torch.zeros((G, P), dtype=torch.float32, device=gt_t.device)
    
    gt_f = gt_t.reshape(G, -1).to(torch.float32)   # [G,HW]
    pr_f = pr_t.reshape(P, -1).to(torch.float32)   # [P,HW]
    inter = gt_f @ pr_f.T                          # [G,P]
    area_g = gt_f.sum(dim=1)                       # [G]
    area_p = pr_f.sum(dim=1)                       # [P]
    union = area_g[:, None] + area_p[None, :] - inter
    return inter / (union + 1e-10)


def _boxes_from_stack_stable(mstk):
    """
    Extract bounding boxes from stacked masks.
    mstk: [N,H,W] bool on device -> boxes [N,4] float32 (x1,y1,x2,y2)
    """
    N, H, W = mstk.shape
    boxes = torch.zeros((N, 4), dtype=torch.float32, device=mstk.device)
    if N == 0:
        return boxes
    
    # y bounds
    rows_any = mstk.any(dim=2)  # [N,H]
    y_idx = torch.arange(H, device=mstk.device)
    y_min = torch.where(rows_any, y_idx, H).amin(dim=1)
    y_max = torch.where(rows_any, y_idx, -1).amax(dim=1)
    
    # x bounds
    cols_any = mstk.any(dim=1)  # [N,W]
    x_idx = torch.arange(W, device=mstk.device)
    x_min = torch.where(cols_any, x_idx, W).amin(dim=1)
    x_max = torch.where(cols_any, x_idx, -1).amax(dim=1)
    
    # clamp empty masks to zeros
    y_min = torch.clamp_min(y_min, 0); y_max = torch.clamp_min(y_max, 0)
    x_min = torch.clamp_min(x_min, 0); x_max = torch.clamp_min(x_max, 0)
    
    boxes[:, 0] = x_min.to(torch.float32)
    boxes[:, 1] = y_min.to(torch.float32)
    boxes[:, 2] = x_max.to(torch.float32)
    boxes[:, 3] = y_max.to(torch.float32)
    return boxes


def _auto_pred_chunk(g_blk, H, W, free_bytes, cap=4096):
    """
    Adaptively determine pred chunk size based on available VRAM.
    Solves: 4*(g*HW + p*HW + g*p) < budget -> p < (B/4 - g*HW) / (HW + g)
    """
    if not ADAPTIVE_P:
        return CHUNK_P
    
    B = int(free_bytes * 0.6)  # 60% budget for safety
    HW = H * W
    num = (B // 4) - (g_blk * HW)
    den = (HW + g_blk)
    if den <= 0 or num <= 0:
        return max(64, min(CHUNK_P, cap))
    
    p_est = max(64, min(int(num // den), cap))
    # align to 32 for better kernel performance
    p_est = int(max(32, (p_est // 32) * 32))
    return max(64, p_est)


# =====================
# GPU Evaluator
# =====================
def compute_metrics_gpu(
    gt_anns: List[Dict[str, Any]],
    pred_anns: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_thr: float = None,
    downscale: int = 1
) -> Dict[str, float]:
    """
    GPU-accelerated evaluation metrics computation.
    Uses vectorized mask IoU via matmul with adaptive chunking for VRAM management.
    
    Args:
        gt_anns: Ground truth annotations
        pred_anns: Prediction annotations
        image_shape: (H, W) tuple
        iou_thr: IoU threshold for matching (default: IOU_THRESHOLD)
        downscale: Downscale factor for masks (default: 1)
    
    Returns:
        Dictionary with metrics: Precision, Recall, F1 Score, Dice Coefficient,
        Avg Fragments, Count Error, PQ, Mean Error
    """
    if iou_thr is None:
        iou_thr = IOU_THRESHOLD
    
    if not USE_CUDA or torch is None:
        # Fallback to CPU version
        return compute_metrics_cpu(gt_anns, pred_anns, image_shape, iou_thr, downscale)
    
    # Build masks on CPU
    gt_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in gt_anns]
    pr_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in pred_anns]
    gt_cls = ensure_class_ids(gt_anns)
    pr_cls = ensure_class_ids(pred_anns)
    
    # Group by class
    gt_by, pr_by = defaultdict(list), defaultdict(list)
    for i, c in enumerate(gt_cls):
        gt_by[c].append(i)
    for j, c in enumerate(pr_cls):
        pr_by[c].append(j)
    
    TP = 0
    total_iou = 0.0
    matched_gt, matched_pr = set(), set()
    fragments = []
    mean_err_acc = []
    
    # Adjust image shape for downscaling
    eval_shape = image_shape if downscale == 1 else (image_shape[0] // downscale, image_shape[1] // downscale)
    
    # Per-class evaluation
    for c in set(gt_by) | set(pr_by):
        gi = gt_by.get(c, [])
        pj = pr_by.get(c, [])
        G, P = len(gi), len(pj)
        if G == 0:
            continue
        
        # Transfer GT masks to GPU with pinned memory
        gt_np = np.stack([gt_masks[k] for k in gi], 0).astype(np.bool_)
        gt_t = torch.from_numpy(gt_np).pin_memory() if gt_np.size else torch.empty((0, *eval_shape), dtype=torch.bool)
        gt_t = gt_t.to(DEVICE, non_blocking=True)
        
        frag = torch.zeros(G, dtype=torch.int64, device=DEVICE)
        
        if P > 0:
            pr_np = np.stack([pr_masks[k] for k in pj], 0).astype(np.bool_)
            pr_t_full = torch.from_numpy(pr_np).pin_memory() if pr_np.size else torch.empty((0, *eval_shape), dtype=torch.bool)
            pr_t_full = pr_t_full.to(DEVICE, non_blocking=True)
            
            H, W = gt_t.shape[-2], gt_t.shape[-1]
            
            # Best IoU per pred
            best_iou = torch.zeros(P, dtype=torch.float32, device=DEVICE)
            best_gt = torch.full((P,), -1, dtype=torch.long, device=DEVICE)
            
            # Chunked processing to fit VRAM
            for gs in range(0, G, CHUNK_G):
                ge = min(gs + CHUNK_G, G)
                gt_blk = gt_t[gs:ge]  # [g',H,W]
                
                # Adaptive pred chunk from free VRAM
                p_step = CHUNK_P
                if DEVICE.type == "cuda" and torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    p_step = _auto_pred_chunk(ge - gs, H, W, free_b, cap=max(1024, CHUNK_P))
                
                for ps in range(0, P, p_step):
                    pe = min(ps + p_step, P)
                    pr_blk = pr_t_full[ps:pe]  # [p',H,W]
                    
                    iou_blk = _pairwise_iou_masks_torch(gt_blk, pr_blk)  # [g',p']
                    vals, arg = torch.max(iou_blk, dim=0)  # per pred in chunk
                    cur = best_iou[ps:pe]
                    upd = vals > cur
                    if upd.any():
                        cur[upd] = vals[upd]
                        best_gt[ps:pe][upd] = arg[upd] + gs
            
            # Match above threshold
            keep = best_iou >= iou_thr
            if keep.any():
                sel_pr_local = torch.nonzero(keep, as_tuple=False).squeeze(1)
                sel_gt_local = best_gt[keep]
                
                frag += torch.bincount(sel_gt_local, minlength=G)
                TP += int(keep.sum().item())
                total_iou += float(best_iou[keep].sum().item())
                
                for jj in sel_pr_local.tolist():
                    matched_pr.add(pj[jj])
                for ii in sel_gt_local.unique().tolist():
                    matched_gt.add(gi[ii])
                
                # Mean Error from boxes
                gt_boxes = _boxes_from_stack_stable(gt_t)  # [G,4]
                pr_boxes = _boxes_from_stack_stable(pr_t_full)  # [P,4]
                gt_xy = gt_boxes[sel_gt_local, :2]
                pr_xy = pr_boxes[sel_pr_local, :2]
                if gt_xy.numel():
                    err = torch.linalg.norm(gt_xy - pr_xy, dim=1)
                    mean_err_acc.extend(err.detach().cpu().tolist())
            
            del pr_t_full
        
        fragments.extend(frag.detach().cpu().tolist())
        del gt_t, frag
    
    # Clean up GPU memory
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Compute final metrics
    FP = len(pr_masks) - len(matched_pr)
    FN = len(gt_masks) - len(matched_gt)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    dice = 2 * total_iou / (TP + 1e-10)
    avg_frag = float(np.mean([f for f in fragments if f > 0])) if fragments else 0.0
    count_err = abs(len(pr_masks) - len(gt_masks))
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-10)
    sq = total_iou / (TP + 1e-10)
    pq = dq * sq
    mean_err = float(np.mean(mean_err_acc)) if mean_err_acc else 0.0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Dice Coefficient": dice,
        "Avg Fragments": avg_frag,
        "Count Error": count_err,
        "PQ": pq,
        "Mean Error": mean_err
    }


# =====================
# CPU Fallback
# =====================
def compute_metrics_cpu(
    gt_anns: List[Dict[str, Any]],
    pred_anns: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_thr: float = None,
    downscale: int = 1
) -> Dict[str, float]:
    """
    CPU fallback evaluator. Matches GPU version output for consistency.
    Simple greedy IoU matching algorithm.
    """
    if iou_thr is None:
        iou_thr = IOU_THRESHOLD
    
    gt_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in gt_anns]
    pr_masks = [load_mask_from_segmentation(a.get("segmentation", []), image_shape, downscale) for a in pred_anns]
    gt_cls = ensure_class_ids(gt_anns)
    pr_cls = ensure_class_ids(pred_anns)
    
    matched_gt, matched_pr = set(), set()
    matched_pairs = []
    TP = 0
    total_iou = 0.0
    fragments = []
    
    for i, gm in enumerate(gt_masks):
        matched = 0
        for j, pm in enumerate(pr_masks):
            if pr_cls[j] != gt_cls[i] or j in matched_pr:
                continue
            inter = np.logical_and(gm, pm).sum(dtype=np.int64)
            union = np.logical_or(gm, pm).sum(dtype=np.int64)
            iou = float(inter) / (float(union) + 1e-10)
            if iou >= iou_thr:
                matched += 1
                matched_gt.add(i)
                matched_pr.add(j)
                matched_pairs.append((i, j))
                TP += 1
                total_iou += iou
        fragments.append(matched)
    
    FP = len(pr_masks) - len(matched_pr)
    FN = len(gt_masks) - len(matched_gt)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    dice = 2 * total_iou / (TP + 1e-10)
    avg_frag = float(np.mean([f for f in fragments if f > 0])) if fragments else 0.0
    count_err = abs(len(pr_masks) - len(gt_masks))
    dq = TP / (TP + 0.5*FP + 0.5*FN + 1e-10)
    sq = total_iou / (TP + 1e-10)
    pq = dq * sq
    
    # Compute Mean Error (bbox centroid distance)
    mean_err = 0.0
    if matched_pairs:
        errors = []
        for i, j in matched_pairs:
            ys_g, xs_g = np.where(gt_masks[i])
            ys_p, xs_p = np.where(pr_masks[j])
            if xs_g.size > 0 and xs_p.size > 0:
                gt_center = np.array([xs_g.min(), ys_g.min()], dtype=np.float32)
                pr_center = np.array([xs_p.min(), ys_p.min()], dtype=np.float32)
                error = np.linalg.norm(gt_center - pr_center)
                errors.append(float(error))
        mean_err = float(np.mean(errors)) if errors else 0.0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Dice Coefficient": dice,
        "Avg Fragments": avg_frag,
        "Count Error": count_err,
        "PQ": pq,
        "Mean Error": mean_err
    }
