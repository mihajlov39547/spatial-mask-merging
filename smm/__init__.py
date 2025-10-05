# smm/__init__.py
# Public package API for Spatial Mask Merging (SMM).

from .smm import SpatialMaskMerger, smm_merge

__all__ = [
    "SpatialMaskMerger",
    "smm_merge",
    "SMMPrediction",
    "SMMAnnotation",
]

__version__ = "0.1.0"
