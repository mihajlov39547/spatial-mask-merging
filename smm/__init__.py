# smm/__init__.py
# Public package API for Spatial Mask Merging (SMM).

from .smm import SpatialMaskMerger, smm_merge
from .predictions import SMMPrediction, SMMAnnotation

__all__ = [
    "SpatialMaskMerger",
    "smm_merge",
    "SMMPrediction",
    "SMMAnnotation",
]

__version__ = "0.1.0"

# Clean up namespace: remove internal module references
# This prevents pollution of the public API with implementation details
import sys as _sys
_current_module = _sys.modules[__name__]

# Remove submodule references from the package namespace
for _attr in ['smm', 'predictions', 'rtree_utils']:
    if hasattr(_current_module, _attr):
        delattr(_current_module, _attr)

# Clean up temporary variables
del _sys, _current_module, _attr
