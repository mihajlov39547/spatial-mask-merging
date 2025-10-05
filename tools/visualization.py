# visualization.py
# Spatial Mask Merge — visualization utilities
# Copyright (c) 2025 Marko Mihajlović
# with contributions from Marina Marjanović
# License: MIT

import os
import sys
import json
import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration / Constants
# =========================

# BGR for OpenCV drawing
CLASS_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (127, 0, 0),
    1: (127, 127, 0),
    2: (255, 127, 0),
    3: (63, 63, 0),
    4: (63, 0, 0),
    5: (155, 100, 0),
    6: (255, 0, 0),
    7: (191, 127, 0),
    8: (127, 63, 0),
    9: (255, 63, 0),
    10: (0, 63, 0),
    11: (63, 127, 0),
    12: (191, 63, 0),
    13: (127, 191, 0),
    14: (191, 0, 0),
}

# Optional class-name lookup (safe defaults)
CLASS_NAME_MAP: Dict[int, str] = {
    0: "small_vehicle",
    1: "large_vehicle",
    2: "plane",
    3: "storage_tank",
    4: "ship",
    5: "harbor",
    6: "swimming_pool",
    11: "bridge",
    13: "roundabout",
    14: "helicopter",
    7: "sport_court",
    8: "sport_court",
    9: "sport_court",
    10: "sport_court",
    12: "sport_court",
}

# =========================
# Utils
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _bgr_to_rgba(color_bgr: Tuple[int, int, int], alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert BGR (0-255) to RGBA in 0-1 range for matplotlib patches if needed."""
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0, float(np.clip(alpha, 0.0, 1.0)))


def _polygon_to_cv(pts_xy: Sequence[Tuple[float, float]]) -> np.ndarray:
    """[(x,y), ...] -> OpenCV int32 Nx1x2"""
    if len(pts_xy) == 0:
        return np.empty((0, 1, 2), dtype=np.int32)
    arr = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    return arr.astype(np.int32).reshape(-1, 1, 2)


def _draw_poly(
    img_bgr: np.ndarray,
    polygon_xy: Sequence[Tuple[float, float]],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> None:
    """Draw a polygon (optional filled) on BGR image."""
    poly_cv = _polygon_to_cv(polygon_xy)
    if poly_cv.size == 0:
        return

    if fill_alpha and fill_alpha > 0.0:
        # Filled overlay via mask blending to avoid matplotlib dependency for fill
        overlay = img_bgr.copy()
        cv2.fillPoly(overlay, [poly_cv], color=color_bgr)
        img_bgr[:] = cv2.addWeighted(overlay, float(np.clip(fill_alpha, 0.0, 1.0)), img_bgr, 1.0 - float(np.clip(fill_alpha, 0.0, 1.0)), 0)

    cv2.polylines(img_bgr, [poly_cv], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


def _put_label(
    img_bgr: np.ndarray,
    text: str,
    anchor_xy: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    x, y = int(anchor_xy[0]), int(anchor_xy[1])
    cv2.putText(
        img_bgr,
        text,
        (x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _save_fig_bgr(img_bgr: np.ndarray, out_path: str, dpi: int = 150) -> None:
    """Save BGR image as PDF/PNG via matplotlib (handles DPI, no axes)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _ensure_dir(out_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close()


# =========================
# Ground Truth (YOLO-style .txt with normalized x y ...)
# =========================

def draw_gt_polygons_on_image(
    image_path: str,
    label_path: str,
    output_path: str,
    class_color_map: Dict[int, Tuple[int, int, int]] = CLASS_COLOR_MAP,
    class_name_map: Dict[int, str] = CLASS_NAME_MAP,
    thickness: int = 2,
    fill_alpha: float = 0.0,
    font_scale: float = 0.6,
) -> None:
    """
    Label format per line:
      <class_id> x1 y1 x2 y2 ... (normalized in [0,1], pairs for polygon)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    h, w = img.shape[:2]

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3 or len(parts) % 2 == 0:
                    # not enough coords, or not pairs
                    continue

                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                # convert normalized coords to absolute pixels
                points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]

                if len(points) >= 3:
                    # optional convex hull for safety (match your previous code)
                    pts_np = cv2.convexHull(np.array(points, dtype=np.int32)).reshape(-1, 2)
                else:
                    pts_np = np.array(points, dtype=np.int32)

                color = class_color_map.get(class_id, (255, 255, 255))
                _draw_poly(img, pts_np.tolist(), color, thickness=thickness, fill_alpha=fill_alpha)

                label = class_name_map.get(class_id, f"{class_id}")
                if len(points) > 0:
                    _put_label(img, label, points[0], color, font_scale=font_scale, thickness=max(1, thickness - 1))
    except FileNotFoundError:
        print(f"⚠️ Label file not found for {image_path}")
        return

    _save_fig_bgr(img, output_path)
    print(f"✅ Saved GT: {output_path}")


def process_gt_folder(
    folder_path: str,
    suffix_image: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    out_suffix: str = "_gt_visualization.pdf",
    **kwargs,
) -> None:
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(suffix_image):
            continue
        base = os.path.splitext(fname)[0]
        image_path = os.path.join(folder_path, fname)
        label_path = os.path.join(folder_path, base + ".txt")
        output_pdf = os.path.join(folder_path, base + out_suffix)
        draw_gt_polygons_on_image(image_path, label_path, output_pdf, **kwargs)


# =========================
# Predictions (JSON)
# Expected JSON structure:
# {
#   "image_name": "xxx.png",
#   "annotations": [
#       {
#         "class_id": int,
#         "type": "small_vehicle",     # optional
#         "segmentation": [[x1, y1, x2, y2, ...], ...]  # one or many polygons (abs pixel coords)
#         # optionally: "bbox": [x,y,w,h] if no segmentation
#       }, ...
#   ]
# }
# =========================

def draw_prediction_json(
    json_path: str,
    image_dir: str,
    output_path: Optional[str] = None,
    class_color_map: Dict[int, Tuple[int, int, int]] = CLASS_COLOR_MAP,
    class_name_map: Dict[int, str] = CLASS_NAME_MAP,
    thickness: int = 2,
    fill_alpha: float = 0.0,
    font_scale: float = 0.6,
) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data.get("image_name")
    annotations = data.get("annotations", [])
    if not image_name:
        print(f"❌ 'image_name' missing in JSON: {json_path}")
        return

    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"❌ Image not found for: {image_name}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    for ann in annotations:
        class_id = int(ann.get("class_id", -1))
        color = class_color_map.get(class_id, (255, 255, 255))

        seg: List[List[float]] = ann.get("segmentation", [])
        polygons: List[List[Tuple[float, float]]] = []
        # segmentation could be [poly], or multiple polys
        for poly in seg:
            if not poly:
                continue
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            polygons.append([(float(x), float(y)) for x, y in arr])

        if not polygons:
            # fallback to bbox if provided (x,y,w,h) -> rectangle polygon
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                polygons = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]

        for poly in polygons:
            if len(poly) < 3:
                continue
            _draw_poly(img, poly, color, thickness=thickness, fill_alpha=fill_alpha)
            # label anchor at first vertex
            x0, y0 = map(int, poly[0])
            label_txt = ann.get("type", class_name_map.get(class_id, str(class_id)))
            _put_label(img, label_txt, (x0, y0), color, font_scale=font_scale, thickness=max(1, thickness - 1))

    if output_path is None:
        output_path = os.path.splitext(json_path)[0] + "_pred_visualization.pdf"

    _save_fig_bgr(img, output_path)
    print(f"✅ Saved Pred: {output_path}")


def process_prediction_folders(
    pred_base_dir: str,
    image_dir: str,
    recursive: bool = True,
    out_ext: str = ".pdf",
    **kwargs,
) -> None:
    for root, dirs, files in os.walk(pred_base_dir):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            out_path = os.path.splitext(json_path)[0] + f"_pred_visualization{out_ext}"
            draw_prediction_json(json_path, image_dir, output_path=out_path, **kwargs)
        if not recursive:
            break


# =========================
# Compare (overlay GT + Preds)
# =========================

def draw_compare_overlay(
    image_path: str,
    gt_label_path: Optional[str],
    pred_json_path: Optional[str],
    image_dir_for_pred: Optional[str] = None,
    output_path: Optional[str] = None,
    gt_alpha: float = 0.25,
    pred_alpha: float = 0.45,
    thickness: int = 2,
    font_scale: float = 0.6,
    gt_color_map: Dict[int, Tuple[int, int, int]] = CLASS_COLOR_MAP,
    pred_color_map: Dict[int, Tuple[int, int, int]] = CLASS_COLOR_MAP,
    class_name_map: Dict[int, str] = CLASS_NAME_MAP,
) -> None:
    """
    Overlay both GT and predictions with different alpha values.
    - GT drawn first (lighter), then predictions (stronger).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    h, w = img.shape[:2]

    # --- Draw GT ---
    if gt_label_path and os.path.exists(gt_label_path):
        with open(gt_label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3 or len(parts) % 2 == 0:
                    continue
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
                if len(points) >= 3:
                    pts_np = cv2.convexHull(np.array(points, dtype=np.int32)).reshape(-1, 2)
                else:
                    pts_np = np.array(points, dtype=np.int32)
                color = gt_color_map.get(class_id, (255, 255, 255))
                _draw_poly(img, pts_np.tolist(), color, thickness=thickness, fill_alpha=gt_alpha)
                if len(points) > 0:
                    _put_label(img, f"GT:{class_name_map.get(class_id, class_id)}", points[0], color, font_scale=font_scale, thickness=max(1, thickness - 1))
    else:
        if gt_label_path:
            print(f"⚠️ GT label file not found: {gt_label_path}")

    # --- Draw Predictions ---
    if pred_json_path:
        with open(pred_json_path, "r") as f:
            pdata = json.load(f)
        image_name = pdata.get("image_name")
        if image_dir_for_pred and image_name:
            img_for_pred = os.path.join(image_dir_for_pred, image_name)
            if os.path.abspath(img_for_pred) != os.path.abspath(image_path):
                print("⚠️ Pred JSON image_name differs from provided image_path; continuing with provided image.")
        anns = pdata.get("annotations", [])
        for ann in anns:
            class_id = int(ann.get("class_id", -1))
            color = pred_color_map.get(class_id, (255, 255, 255))
            seg = ann.get("segmentation", [])
            polygons = []
            for poly in seg:
                if not poly:
                    continue
                arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                polygons.append([(float(x), float(y)) for x, y in arr])
            if not polygons:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w0, h0 = bbox
                    polygons = [[(x, y), (x + w0, y), (x + w0, y + h0), (x, y + h0)]]
            for poly in polygons:
                if len(poly) < 3:
                    continue
                _draw_poly(img, poly, color, thickness=thickness, fill_alpha=pred_alpha)
                x0, y0 = map(int, poly[0])
                _put_label(img, f"P:{class_name_map.get(class_id, class_id)}", (x0, y0), color, font_scale=font_scale, thickness=max(1, thickness - 1))

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(image_path), base + "_compare.pdf")

    _save_fig_bgr(img, output_path)
    print(f"✅ Saved Compare: {output_path}")


# =========================
# CLI
# =========================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spatial Mask Merge — visualization helpers for GT, predictions, and compare overlays."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # GT
    gt = sub.add_parser("gt", help="Draw ground-truth polygons from YOLO-style .txt files in a folder.")
    gt.add_argument("--folder", required=True, help="Folder containing images and .txt labels with same basename.")
    gt.add_argument("--out-suffix", default="_gt_visualization.pdf", help="Output file suffix.")
    gt.add_argument("--fill-alpha", type=float, default=0.0, help="Fill opacity for GT polygons (0-1).")
    gt.add_argument("--thickness", type=int, default=2, help="Polygon edge thickness.")
    gt.add_argument("--font-scale", type=float, default=0.6, help="Label font scale.")

    # Preds
    pr = sub.add_parser("preds", help="Draw prediction polygons from JSON files under a base directory.")
    pr.add_argument("--pred-base-dir", required=True, help="Base directory containing JSON files (recursive).")
    pr.add_argument("--image-dir", required=True, help="Directory with source images referenced by JSON 'image_name'.")
    pr.add_argument("--no-recursive", action="store_true", help="Disable recursion.")
    pr.add_argument("--out-ext", default=".pdf", choices=[".pdf", ".png"], help="Output image extension.")
    pr.add_argument("--fill-alpha", type=float, default=0.0, help="Fill opacity for prediction polygons (0-1).")
    pr.add_argument("--thickness", type=int, default=2, help="Polygon edge thickness.")
    pr.add_argument("--font-scale", type=float, default=0.6, help="Label font scale.")

    # Compare
    cp = sub.add_parser("compare", help="Overlay GT and predictions on the same image.")
    cp.add_argument("--image-path", required=True, help="Path to the image to visualize.")
    cp.add_argument("--gt-label", default=None, help="Path to GT .txt label file for the image (optional).")
    cp.add_argument("--pred-json", default=None, help="Path to predictions JSON for the image (optional).")
    cp.add_argument("--pred-image-dir", default=None, help="Image dir referred by pred JSON (optional).")
    cp.add_argument("--out", default=None, help="Explicit output path (.pdf/.png).")
    cp.add_argument("--gt-alpha", type=float, default=0.25, help="Fill opacity for GT polygons (0-1).")
    cp.add_argument("--pred-alpha", type=float, default=0.45, help="Fill opacity for prediction polygons (0-1).")
    cp.add_argument("--thickness", type=int, default=2, help="Polygon edge thickness.")
    cp.add_argument("--font-scale", type=float, default=0.6, help="Label font scale.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.cmd == "gt":
        process_gt_folder(
            folder_path=args.folder,
            out_suffix=args.out_suffix,
            fill_alpha=args.fill_alpha,
            thickness=args.thickness,
            font_scale=args.font_scale,
        )

    elif args.cmd == "preds":
        process_prediction_folders(
            pred_base_dir=args.pred_base_dir,
            image_dir=args.image_dir,
            recursive=(not args.no_recursive),
            out_ext=args.out_ext,
            fill_alpha=args.fill_alpha,
            thickness=args.thickness,
            font_scale=args.font_scale,
        )

    elif args.cmd == "compare":
        draw_compare_overlay(
            image_path=args.image_path,
            gt_label_path=args.gt_label,
            pred_json_path=args.pred_json,
            image_dir_for_pred=args.pred_image_dir,
            output_path=args.out,
            gt_alpha=args.gt_alpha,
            pred_alpha=args.pred_alpha,
            thickness=args.thickness,
            font_scale=args.font_scale,
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
