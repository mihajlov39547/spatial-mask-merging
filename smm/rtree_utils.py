# smm/rtree_utils.py
# Minimal R-tree wrapper with a graceful pure-Python fallback.
# This module provides a uniform interface used by the SMM pipeline:
#   - RTreeIndex.insert(i, bbox)
#   - RTreeIndex.query(query_bbox) -> iterable of integer ids
#
# The primary implementation uses the 'rtree' package (libspatialindex).
# If 'rtree' is not available, a pure-Python fallback will be used that
# performs O(N) axis-aligned rectangle intersection checks. This preserves
# correctness but will be slower on large inputs.
#
# bbox format throughout: (x1, y1, x2, y2) with x1 <= x2, y1 <= y2

from __future__ import annotations
from typing import Iterable, Tuple, List
import warnings

try:
    from rtree import index as _rtree_index
    _RTREE_AVAILABLE = True
except Exception:
    _RTREE_AVAILABLE = False
    # Warn user that fallback mode is being used
    warnings.warn(
        "rtree library not found. Using pure-Python fallback with O(N) query performance. "
        "For better performance on large datasets, install: pip install rtree",
        RuntimeWarning,
        stacklevel=2
    )


BBox = Tuple[float, float, float, float]


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    """Axis-aligned rectangle intersection test (inclusive edges)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # Single compound check (micro-optimization: fewer branches)
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


class RTreeIndex:
    """
    Unified index API used by SMM.

    Methods
    -------
    insert(i: int, bbox: BBox) -> None
        Insert an item id and its bounding box.
    query(bbox: BBox) -> Iterable[int]
        Return ids whose rectangles intersect the query rectangle.
    is_optimized() -> bool
        Returns True if using rtree library, False if using fallback.

    Notes
    -----
    - With the 'rtree' package available, queries run in sublinear time.
    - Without it, a simple list-based fallback is used (O(N) scan).
    """
    def __init__(self) -> None:
        if _RTREE_AVAILABLE:
            # Configure libspatialindex-backed structure
            p = _rtree_index.Property()
            # Depending on workloads, these can be tuned:
            # p.fill_factor = 0.9
            # p.index_capacity = 100
            # p.leaf_capacity = 100
            self._use_fallback = False
            self._idx = _rtree_index.Index(properties=p)
        else:
            # Fallback storage: list of (id, bbox)
            self._use_fallback = True
            self._items: List[Tuple[int, BBox]] = []

    def insert(self, i: int, bbox: BBox) -> None:
        """Insert an item with its bounding box into the spatial index."""
        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (TypeError, ValueError, IndexError) as e:
            raise TypeError(f"Invalid bbox format: expected 4 numeric values, got {bbox}") from e
        
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox geometry: expected x1<=x2 and y1<=y2, got ({x1}, {y1}, {x2}, {y2})")
        
        if self._use_fallback:
            self._items.append((i, (x1, y1, x2, y2)))
        else:
            self._idx.insert(i, (x1, y1, x2, y2))

    def query(self, bbox: BBox) -> Iterable[int]:
        """
        Return ids whose rectangles intersect the query bbox (inclusive edges).
        This returns a superset of L2-within-ρ neighbors; downstream code
        should still apply any exact geometric post-filter (e.g., L2 distance).
        """
        try:
            qx1, qy1, qx2, qy2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (TypeError, ValueError, IndexError) as e:
            raise TypeError(f"Invalid query bbox format: expected 4 numeric values, got {bbox}") from e
        
        if qx2 < qx1 or qy2 < qy1:
            raise ValueError(f"Invalid query bbox geometry: expected x1<=x2 and y1<=y2, got ({qx1}, {qy1}, {qx2}, {qy2})")
        
        if self._use_fallback:
            for i, bb in self._items:
                if _bbox_intersects(bb, (qx1, qy1, qx2, qy2)):
                    yield i
        else:
            # rtree returns iterator of ids intersecting the query rectangle
            yield from self._idx.intersection((qx1, qy1, qx2, qy2))
    
    def is_optimized(self) -> bool:
        """Return True if using rtree library backend, False if using pure-Python fallback."""
        return not self._use_fallback
    
    def __len__(self) -> int:
        """Return the number of items in the index (only available for fallback mode)."""
        if self._use_fallback:
            return len(self._items)
        else:
            # rtree doesn't expose size efficiently
            # Raise NotImplementedError to indicate this operation is not supported
            raise NotImplementedError("Size information not available when using rtree backend")
