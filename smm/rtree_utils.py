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

try:
    from rtree import index as _rtree_index
    _RTREE_AVAILABLE = True
except Exception:
    _RTREE_AVAILABLE = False


BBox = Tuple[float, float, float, float]


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    """Axis-aligned rectangle intersection test (inclusive edges)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < bx1 or bx2 < ax1:
        return False
    if ay2 < by1 or by2 < ay1:
        return False
    return True


class RTreeIndex:
    """
    Unified index API used by SMM.

    Methods
    -------
    insert(i: int, bbox: BBox) -> None
        Insert an item id and its bounding box.
    query(bbox: BBox) -> Iterable[int]
        Return ids whose rectangles intersect the query rectangle.

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
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid bbox: expected x1<=x2 and y1<=y2.")
        if self._use_fallback:
            self._items.append((i, (float(x1), float(y1), float(x2), float(y2))))
        else:
            self._idx.insert(i, (float(x1), float(y1), float(x2), float(y2)))

    def query(self, bbox: BBox) -> Iterable[int]:
        """
        Return ids whose rectangles intersect the query bbox (inclusive edges).
        This returns a superset of L2-within-ρ neighbors; downstream code
        should still apply any exact geometric post-filter (e.g., L2 distance).
        """
        qx1, qy1, qx2, qy2 = bbox
        if self._use_fallback:
            for i, bb in self._items:
                if _bbox_intersects(bb, (qx1, qy1, qx2, qy2)):
                    yield i
        else:
            # rtree returns iterator of ids intersecting the query rectangle
            yield from self._idx.intersection((qx1, qy1, qx2, qy2))
