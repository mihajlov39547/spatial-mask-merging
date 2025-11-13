"""
SMM Hyperparameter Optimizer (Optuna)
====================================

Bayesian optimization of Spatial Mask Merging (SMM) hyperparameters using Optuna.

Adapts the SMM optimizer to the repository structure:
- Uses `smm.smm.SpatialMaskMerger` as the backend (ILP-based correlation clustering).
- Converts polygon annotations to binary masks via SMMPrediction container.
- Evaluates with a simple F1 metric (greedy IoU matching).
- Saves best params and feature importances.

Place this file at: `tools/optimize_smm.py` (or anywhere you prefer).

Requirements
------------
pip install optuna numpy pandas tqdm opencv-python-headless networkx matplotlib pulp rtree
(Optionally) pip install torch rtree pulp

Note: CUDA/GPU acceleration is not used in SMM optimization itself (the ILP solver and 
mask operations run on CPU). GPU is only beneficial for the separate evaluation script.

Usage
-----
python tools/optimize_smm.py --pred_dir <dir-with-pred-json> --gt_dir <dir-with-gt-json> \
  --img_dir <dir-with-images> --out_dir ./opt_results --trials 30
"""

from __future__ import annotations

# --- Standard library ---
import argparse
import gc
import json
import math
import os
import time
import tracemalloc
from glob import glob
from typing import Dict, List, Tuple, Any

# --- Third-party ---
import cv2
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

# --- Optional (not used in SMM optimization, only for external evaluation) ---
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    USE_CUDA = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- Repo imports ---
# Expecting these modules in your repository:
#   smm.smm -> SpatialMaskMerger class
#   smm.predictions -> SMMPrediction dataclass/container
from smm.smm import SpatialMaskMerger
from smm.predictions import SMMPrediction


# =====================
# CONFIG (overridden by CLI)
# =====================
DEFAULTS = dict(
    method="smm",            # identifier tag
    mode="ilp",              # always "ilp" (SpatialMaskMerger uses ILP)
    subset_per_trial=0,      # 0 = use all files
    n_trials=30,             # number of Optuna trials
)


# =====================
# UTILS
# =====================
def _collect_dir_jsons(d: str) -> List[str]:
    return sorted(glob(os.path.join(d, "**", "*.json"), recursive=True))


def get_image_shape_from_disk(img_dir: str, image_name: str) -> Tuple[int, int]:
    img_path = os.path.join(img_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img.shape[:2]
    return (h, w)


def polygons_to_mask(polygons: List[List[List[float]]], image_size: Tuple[int, int]) -> np.ndarray:
    """polygons: list of lists of [x,y] points; image_size: (H,W)"""
    mask = np.zeros(image_size, dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_to_polygons(mask: np.ndarray, epsilon_frac: float = 0.01) -> List[List[List[int]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon_frac * max(peri, 1.0), True)
        pts = approx.squeeze()
        if pts.ndim == 2 and len(pts) >= 3:
            polygons.append(pts.astype(int).tolist())
    return polygons


def compute_simple_f1(gt_anns: List[Dict[str, Any]], pred_anns: List[Dict[str, Any]], image_size: Tuple[int, int]) -> Dict[str, float]:
    """Very lightweight matcher: greedy IoU > 0.5 = TP. Returns F1, precision, recall."""
    def mask_from_anns(anns):
        return [polygons_to_mask(a["segmentation"], image_size) for a in anns]

    gtm = mask_from_anns(gt_anns)
    prm = mask_from_anns(pred_anns)

    used_p = set()
    tp = 0
    for gm in gtm:
        match_j = -1
        best_iou = 0.0
        for j, pm in enumerate(prm):
            if j in used_p:
                continue
            inter = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            iou = (inter / union) if union > 0 else 0.0
            if iou > 0.5 and iou > best_iou:
                best_iou = iou
                match_j = j
        if match_j >= 0:
            tp += 1
            used_p.add(match_j)

    fp = max(0, len(prm) - len(used_p))
    fn = max(0, len(gtm) - tp)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"F1 Score": float(f1), "Precision": float(precision), "Recall": float(recall)}


# =====================
# SMM RUNNER
# =====================
def run_smm_on_entry(entry: Dict[str, Any], img_dir: str, mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert entry annotations -> SMMPrediction, run SMM, return merged entry with polygons."""
    image_name = entry.get("image_name")
    img_path = os.path.join(img_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    H, W = img.shape[:2]
    image_size = (H, W)

    # Create single SMMPrediction container for the image
    prediction = SMMPrediction(image_name=image_name)
    for ann in entry.get("annotations", []):
        label = ann.get("type", ann.get("label", "object"))
        class_id = ann.get("class_id", 0)
        score = float(ann.get("confidence", ann.get("score", 1.0)))
        seg = ann.get("segmentation", [])
        if not seg:
            continue
        polygons = seg if isinstance(seg[0][0], list) else [seg]
        
        # Add annotation to SMMPrediction
        bbox = ann.get("bbox", [0, 0, 0, 0])
        prediction.add_annotation(
            type=label,
            class_id=class_id,
            confidence=score,
            bbox=tuple(bbox) if len(bbox) == 4 else (0, 0, 0, 0),
            segmentation=polygons
        )

    smm = SpatialMaskMerger(**params)
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.time()

    merged = smm.merge(prediction, image_size_hw=image_size)  # returns list of dicts with mask, bbox, score, label

    exec_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cleaned_annotations = []
    for obj in merged:
        # obj is a dict with keys: mask, bbox, score, label
        poly = mask_to_polygons(obj["mask"])
        if not poly:
            continue
        bbox = obj["bbox"]  # already computed by SMM
        score = float(obj["score"])
        label = obj["label"]
        cleaned_annotations.append({
            "type": str(label),
            "class_id": int(label) if isinstance(label, (int, np.integer)) else 0,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "segmentation": poly,
            "confidence": score
        })

    return dict(
        image_name=image_name,
        image_size=image_size,
        exec_time=exec_time,
        peak_memory=round(peak_mem / 1024, 1),
        annotations=cleaned_annotations
    )


def _get_bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# =====================
# OPTUNA SUGGESTIONS
# =====================
def suggest_params(trial: optuna.trial.Trial, mode: str) -> Dict[str, Any]:
    """
    Hyperparameters for SpatialMaskMerger based on paper parameters.
    
    Parameters match the SMM paper:
    - tau_d: distance threshold (pixels)
    - tau_i: IoU threshold
    - rho: R-tree search radius (pixels)
    - beta1: distance weight in edge weight formula
    - beta2: IoU weight in edge weight formula
    - beta3: confidence weight in edge weight formula
    - gamma: anti-chaining threshold
    - lambda_: correlation clustering penalty
    """
    params = {
        # Spatial thresholds
        "tau_d": trial.suggest_float("tau_d", 5.0, 30.0),
        "tau_i": trial.suggest_float("tau_i", 0.1, 0.9),
        "rho": trial.suggest_float("rho", 10.0, 50.0),
        
        # Edge weight coefficients
        "beta1": trial.suggest_float("beta1", 0.2, 0.4),
        "beta2": trial.suggest_float("beta2", 0.4, 0.6),
        "beta3": trial.suggest_float("beta3", 0.1, 0.3),
        
        # Anti-chaining threshold
        "gamma": trial.suggest_float("gamma", 0.3, 0.7),
        
        # Correlation clustering penalty
        "lambda_": trial.suggest_float("lambda_", 0.1, 2.0, log=True),
    }
    return params


# =====================
# OBJECTIVE
# =====================
def make_objective(pred_dir: str, gt_dir: str, img_dir: str, mode: str, subset_per_trial: int):
    pred_files = _collect_dir_jsons(pred_dir)
    if not pred_files:
        raise RuntimeError(f"No prediction JSON files in: {pred_dir}")

    # intersect on basenames existing in GT
    gt_basenames = {os.path.basename(p) for p in _collect_dir_jsons(gt_dir)}
    pred_files = [p for p in pred_files if os.path.basename(p) in gt_basenames]
    if not pred_files:
        raise RuntimeError("No matching pred/gt basenames.")

    def _objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, mode)
        scores: List[float] = []

        files = pred_files
        if subset_per_trial and len(files) > subset_per_trial:
            files = files[:subset_per_trial]

        for jf in tqdm(files, desc="Trial", leave=False):
            try:
                with open(jf, "r") as f:
                    pred_entry = json.load(f)
                with open(os.path.join(gt_dir, os.path.basename(jf)), "r") as gf:
                    gt_entry = json.load(gf)
            except Exception:
                continue

            try:
                merged = run_smm_on_entry(pred_entry, img_dir, mode, params)
            except Exception:
                continue

            try:
                # Prefer external evaluator if available
                metrics = compute_simple_f1(gt_entry.get("annotations", []),
                                            merged.get("annotations", []),
                                            tuple(merged.get("image_size", (1024, 1024))))
            except Exception:
                continue

            scores.append(float(metrics.get("F1 Score", 0.0)))
            trial.report(float(np.mean(scores)) if scores else 0.0, step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores)) if scores else 0.0

    return _objective


# =====================
# ARTIFACTS
# =====================
def save_importances(study: optuna.study.Study, out_dir: str, tag: str = "smm"):
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    imp = optuna.importance.get_param_importances(study)
    with open(os.path.join(out_dir, f"{tag}_hparam_importance.json"), "w") as f:
        json.dump(imp, f, indent=2)

    labels, values = list(imp.keys()), list(imp.values())
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Relative Importance")
    plt.title("SMM Hyperparameter Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_hparam_importance.pdf"), format="pdf")
    plt.close()


def save_trials_csv(study: optuna.study.Study, out_dir: str, tag: str = "smm"):
    os.makedirs(out_dir, exist_ok=True)
    df = study.trials_dataframe(attrs=("number","value","state","params","datetime_start","datetime_complete","duration"))
    df.to_csv(os.path.join(out_dir, f"{tag}_optuna_trials.csv"), index=False)


# =====================
# MAIN
# =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="Directory with prediction JSON files")
    p.add_argument("--gt_dir", required=True, help="Directory with GT JSON files")
    p.add_argument("--img_dir", required=True, help="Directory with the corresponding raw images (for shape)")
    p.add_argument("--out_dir", default="./opt_results", help="Where to store artifacts")
    p.add_argument("--mode", default=DEFAULTS["mode"], choices=["ilp"], help="SMM backend (only ILP supported)")
    p.add_argument("--subset_per_trial", type=int, default=DEFAULTS["subset_per_trial"])
    p.add_argument("--trials", type=int, default=DEFAULTS["n_trials"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        mode=args.mode,
        subset_per_trial=args.subset_per_trial
    )

    study.optimize(objective, n_trials=args.trials)

    print("\n✅ Optimization finished.")
    print(f"Best F1: {study.best_value:.6f}")
    print("Best Params:", study.best_params)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"best_params_{args.mode}.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    save_importances(study, args.out_dir, tag=f"smm_{args.mode}")
    save_trials_csv(study, args.out_dir, tag=f"smm_{args.mode}")


if __name__ == "__main__":
    main()
