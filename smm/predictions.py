# smm/predictions.py
# Standardized prediction container for Spatial Mask Merging (SMM).
# This class enforces the exact JSON structure and 
# converts to the internal SMM object schema used by the solver.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

# -----------------------------
# Types
# -----------------------------

BBox = Tuple[float, float, float, float]            # (x1, y1, x2, y2)
Point = Tuple[Union[int, float], Union[int, float]]
Polygon = List[Point]                               # [[x,y], [x,y], ...]
Polygons = List[Polygon]
SizeHW = Tuple[int, int]                            # (H, W)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SMMAnnotation:
    """
    One instance prediction (one object).
    This captures the exact fields of your standardized JSON:
      - type        : class name (string)
      - class_id    : class id (int)
      - confidence  : float in [0,1]
      - bbox        : (x1,y1,x2,y2) in pixels (inclusive bounds)
      - segmentation: list of polygons; each polygon is a list of [x,y] pairs
    """
    type: str
    class_id: int
    confidence: float
    bbox: BBox
    segmentation: Polygons

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if not (x2 >= x1 and y2 >= y1):
            raise ValueError(f"Invalid bbox: {self.bbox}. Expected x1<=x2 and y1<=y2.")
        if not (0.0 <= float(self.confidence) <= 1.0):
            # We do not clamp silently; let caller decide
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}.")
        # permit degenerate polygons; they will rasterize to empty masks


@dataclass
class SMMPrediction:
    """
    Standardized per-image prediction record.
    Fields:
      - image_name: file name of the image (string)
      - annotations: list[SMMAnnotation]
    """
    image_name: str
    annotations: List[SMMAnnotation] = field(default_factory=list)

    # ---------------- Creation / I/O ----------------

    def add_annotation(self,
                       type: str,
                       class_id: int,
                       confidence: float,
                       bbox: BBox,
                       segmentation: Polygons) -> None:
        ann = SMMAnnotation(
            type=str(type),
            class_id=int(class_id),
            confidence=float(confidence),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            segmentation=[[(float(x), float(y)) for (x, y) in poly] for poly in segmentation],
        )
        ann.validate()
        self.annotations.append(ann)

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Export to the exact standardized JSON structure.
        """
        return {
            "image_name": self.image_name,
            "annotations": [
                {
                    "type": ann.type,
                    "class_id": int(ann.class_id),
                    "confidence": float(ann.confidence),
                    "bbox": [float(ann.bbox[0]), float(ann.bbox[1]),
                             float(ann.bbox[2]), float(ann.bbox[3])],
                    "segmentation": [
                        [[float(x), float(y)] for (x, y) in poly] for poly in ann.segmentation
                    ],
                }
                for ann in self.annotations
            ],
        }

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> "SMMPrediction":
        """
        Construct from your standardized JSON record.
        """
        image_name = str(d.get("image_name", ""))
        anns_in = d.get("annotations", [])
        anns: List[SMMAnnotation] = []
        for a in anns_in:
            ann = SMMAnnotation(
                type=str(a["type"]),
                class_id=int(a["class_id"]),
                confidence=float(a["confidence"]),
                bbox=(float(a["bbox"][0]), float(a["bbox"][1]),
                      float(a["bbox"][2]), float(a["bbox"][3])),
                segmentation=[
                    [(float(x), float(y)) for (x, y) in poly] for poly in a.get("segmentation", [])
                ],
            )
            ann.validate()
            anns.append(ann)
        return SMMPrediction(image_name=image_name, annotations=anns)

    # ------------- Conversion to SMM core objects -------------

    def to_smm_objects(self,
                       image_size_hw: SizeHW,
                       *,
                       prefer_segmentation: bool = True,
                       recompute_bbox: bool = True,
                       min_polygon_points: int = 3) -> List[Dict[str, Any]]:
        """
        Convert to the internal SMM schema:
          [{"mask": bool(H,W), "bbox": (x1,y1,x2,y2), "score": float, "label": Any}, ...]

        Parameters
        ----------
        image_size_hw       : (H, W) full image size for mask rasterization.
        prefer_segmentation : if True, use polygons when available; otherwise fall back to bbox.
        recompute_bbox      : if True, compute tight bbox from the rasterized mask (recommended).
        min_polygon_points  : polygons with fewer points are ignored (no area).

        Returns
        -------
        list of dicts ready for SpatialMaskMerger.merge(...)
        """
        H, W = int(image_size_hw[0]), int(image_size_hw[1])
        if H <= 0 or W <= 0:
            raise ValueError("to_smm_objects: image_size_hw must be positive (H, W).")

        objs: List[Dict[str, Any]] = []
        for ann in self.annotations:
            # Rasterize to mask
            mask = self._annotation_to_mask(
                ann=ann,
                image_size_hw=(H, W),
                prefer_segmentation=prefer_segmentation,
                min_polygon_points=min_polygon_points,
            )
            bbox = self._tight_bbox_from_mask(mask) if recompute_bbox else ann.bbox
            objs.append({
                "mask": mask,
                "bbox": bbox,
                "score": float(ann.confidence),
                "label": int(ann.class_id),  # use class_id as the canonical label
            })
        return objs

    # ------------- Internals: rasterization & bbox -------------

    @staticmethod
    def _annotation_to_mask(ann: SMMAnnotation,
                            image_size_hw: SizeHW,
                            *,
                            prefer_segmentation: bool,
                            min_polygon_points: int) -> np.ndarray:
        """
        Create a boolean mask (H, W) for one annotation. If segmentation polygons
        are missing or invalid (or prefer_segmentation=False), fallback to bbox.
        """
        H, W = image_size_hw
        if prefer_segmentation and ann.segmentation:
            mask = SMMPrediction._polygons_to_mask(
                polygons=ann.segmentation,
                image_size_hw=(H, W),
                min_polygon_points=min_polygon_points,
            )
            if mask.any():
                return mask
        return SMMPrediction._bbox_to_mask(ann.bbox, (H, W))

    @staticmethod
    def _polygons_to_mask(polygons: Polygons,
                          image_size_hw: SizeHW,
                          min_polygon_points: int = 3) -> np.ndarray:
        """
        Rasterize a set of polygons into a union mask.
        """
        H, W = image_size_hw
        img = Image.new(mode="1", size=(W, H), color=0)
        draw = ImageDraw.Draw(img)
        for poly in polygons:
            if len(poly) < min_polygon_points:
                continue
            pts = [(float(x), float(y)) for (x, y) in poly]
            draw.polygon(pts, outline=1, fill=1)
        return np.array(img, dtype=bool)

    @staticmethod
    def _bbox_to_mask(bbox: BBox, image_size_hw: SizeHW) -> np.ndarray:
        """
        Create a rectangular mask from a bbox (inclusive bounds).
        """
        H, W = image_size_hw
        x1, y1, x2, y2 = bbox
        xi1 = max(0, int(np.floor(x1)))
        yi1 = max(0, int(np.floor(y1)))
        xi2 = min(W - 1, int(np.floor(x2)))
        yi2 = min(H - 1, int(np.floor(y2)))
        mask = np.zeros((H, W), dtype=bool)
        if xi2 >= xi1 and yi2 >= yi1:
            mask[yi1:yi2 + 1, xi1:xi2 + 1] = True
        return mask

    @staticmethod
    def _tight_bbox_from_mask(mask: np.ndarray) -> BBox:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        return (float(x1), float(y1), float(x2), float(y2))
