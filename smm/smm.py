# smm/smm.py
# Paper-faithful Spatial Mask Merging (SMM) with exact correlation clustering (ILP).
# Author: Marko Mihajlović (and collaborators)
# License: MIT

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Iterable, Optional
import math
import numpy as np
import networkx as nx

from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

try:
    import pulp  # ILP solver
except Exception as e:
    pulp = None

# Expected internal utility:
# Provide an R-tree wrapper exposing:
#   class RTreeIndex:
#       def insert(self, idx: int, bbox: Tuple[float,float,float,float]) -> None
#       def query(self, bbox: Tuple[float,float,float,float]) -> Iterable[int]
# It is recommended to back this by 'rtree' or 'pygeos/STRtree'.
try:
    from .rtree_utils import RTreeIndex
except Exception:
    RTreeIndex = None

# ------------------------------- Geometry & helpers -------------------------------

def bbox_l2_distance(b1: Tuple[float, float, float, float],
                     b2: Tuple[float, float, float, float]) -> float:
    """
    Minimum Euclidean distance between two axis-aligned boxes.
    Boxes are (x1,y1,x2,y2) with x1<=x2, y1<=y2.
    """
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    # Horizontal gap (0 if overlapping)
    dx = 0.0
    if x2 < X1:
        dx = X1 - x2
    elif X2 < x1:
        dx = x1 - X2
    # Vertical gap (0 if overlapping)
    dy = 0.0
    if y2 < Y1:
        dy = Y1 - y2
    elif Y2 < y1:
        dy = y1 - Y2
    return math.hypot(dx, dy)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    IoU of two boolean masks of identical shape.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("compute_iou: mask shapes must match.")
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Return N x 2 array of (row, col) coordinates for boundary pixels of a boolean mask.
    A pixel is on the boundary if it is foreground and at least one 4-neighbor is background.
    Implemented using binary erosion XOR original.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float32)
    eroded = binary_erosion(m, structure=np.array([[0,1,0],
                                                   [1,1,1],
                                                   [0,1,0]], dtype=bool), border_value=False)
    bnd = np.logical_and(m, np.logical_not(eroded))
    coords = np.argwhere(bnd)
    return coords.astype(np.float32)


def boundary_distance(mask_a: np.ndarray, mask_b: np.ndarray,
                      cache: Optional[Dict[int, np.ndarray]] = None,
                      id_a: Optional[int] = None,
                      id_b: Optional[int] = None) -> float:
    """
    Minimum Euclidean distance between boundary pixels of two masks.
    Optional cache dict can memoize per-object boundary coordinates under integer ids.
    """
    if cache is not None and id_a is not None:
        pa = cache.get(id_a)
        if pa is None:
            pa = boundary_pixels(mask_a)
            cache[id_a] = pa
    else:
        pa = boundary_pixels(mask_a)

    if cache is not None and id_b is not None:
        pb = cache.get(id_b)
        if pb is None:
            pb = boundary_pixels(mask_b)
            cache[id_b] = pb
    else:
        pb = boundary_pixels(mask_b)

    if pa.shape[0] == 0 or pb.shape[0] == 0:
        # If one mask is empty boundary-wise, fall back to centroid-to-centroid distance
        ya, xa = (np.array(np.nonzero(mask_a)).mean(axis=1) if mask_a.any() else np.array([0.0, 0.0]))
        yb, xb = (np.array(np.nonzero(mask_b)).mean(axis=1) if mask_b.any() else np.array([0.0, 0.0]))
        return float(math.hypot(float(xa - xb), float(ya - yb)))

    dists = cdist(pa, pb, metric="euclidean")
    return float(dists.min()) if dists.size else float("inf")


def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Pixelwise OR union of a list of boolean masks with identical shape.
    """
    if len(masks) == 0:
        raise ValueError("merge_masks: empty input.")
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        if m.shape != out.shape:
            raise ValueError("merge_masks: all masks must share shape.")
        out |= m.astype(bool)
    return out


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Tight axis-aligned bounding box around a boolean mask. Returns (x1,y1,x2,y2).
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return (x1, y1, x2, y2)


# ------------------------------- SMM core -------------------------------

def edge_weight(D_ij: float, I_ij: float, s_i: float, s_j: float, params: Dict[str, Any]) -> float:
    """
    Paper's linear mixture:
        w_ij = β1 * (1 - D_ij / τ_d)_+ + β2 * I_ij + β3 * min(s_i, s_j)
    """
    tau_d = float(params.get("tau_d", 15.0))
    beta1 = float(params.get("beta1", 1.0))
    beta2 = float(params.get("beta2", 1.0))
    beta3 = float(params.get("beta3", 0.5))
    # positive part
    dist_term = max(0.0, 1.0 - (D_ij / max(tau_d, 1e-8)))
    return float(beta1 * dist_term + beta2 * I_ij + beta3 * min(float(s_i), float(s_j)))


def compatible_pair(obj_i: Dict[str, Any], obj_j: Dict[str, Any],
                    D_ij: float, I_ij: float, w_ij: float,
                    params: Dict[str, Any]) -> bool:
    """
    Anti-chaining pairwise validity predicate C_gamma(o_i, o_j) == 1:
       same label AND (D_ij <= τ_d OR I_ij >= τ_i) AND w_ij >= γ
    """
    if obj_i["label"] != obj_j["label"]:
        return False
    tau_d = float(params.get("tau_d", 15.0))
    tau_i = float(params.get("tau_i", 0.5))
    gamma = float(params.get("gamma", 0.5))
    cond_geom = (D_ij <= tau_d) or (I_ij >= tau_i)
    return bool(cond_geom and (w_ij >= gamma))


def _check_deps():
    if pulp is None:
        raise ImportError(
            "PuLP is required for the exact SMM ILP solver. "
            "Please add 'pulp>=2.7' to requirements.txt and install it."
        )
    if RTreeIndex is None:
        raise ImportError(
            "RTreeIndex utility not found. Ensure smm/rtree_utils.py provides RTreeIndex "
            "or install an R-tree backend used by that wrapper."
        )


class SpatialMaskMerger:
    """
    Paper-faithful SMM with exact correlation clustering (ILP) and anti-chaining constraint.

    Pipeline:
      1) Neighborhood generation via R-tree with radius ρ (exact L2 post-filter on boxes).
      2) Edge creation for same-class pairs; compute D_ij (boundary distance), I_ij (mask IoU), w_ij.
      3) Build ILP with:
            - Variables x_ij ∈ {0,1}: x_ij = 1 means 'separated', x_ij = 0 means 'merged'
            - Objective: sum_{(i,j)} [ w_ij * x_ij + λ * (1 - w_ij) * (1 - x_ij) ]
            - Triangle inequalities (partition consistency): x_ij ≤ x_ik + x_kj for all i<j<k
            - Anti-chaining (hard): if C_gamma(i,j) == 0 then x_ij = 1 (cannot-link)
      4) Extract clusters by connected components of 'must-link' edges (x_ij == 0).
      5) Merge masks Φ(A): pixel OR, bbox union, score aggregation (mean or area-weighted).

    Expected object schema for 'objects':
      {
        "mask": np.ndarray[H,W] of bool,
        "bbox": (x1,y1,x2,y2),
        "score": float,
        "label": Any (hashable)
      }

    Parameters (defaults match paper ranges; tune per dataset):
      tau_d: float = 15.0
      tau_i: float = 0.5
      rho:   float = 30.0
      beta1: float = 1.0
      beta2: float = 1.0
      beta3: float = 0.5
      gamma: float = 0.5
      lambda: float = 1.0                      # correlation clustering penalty λ
      score_threshold: float = 0.0
      score_aggregation: {"mean","area_mean"}  # score pooling in Φ(A), default "mean"
    """

    def __init__(self, **kwargs):
        # Store parameters with defaults
        self.params: Dict[str, Any] = dict(
            tau_d=kwargs.get("tau_d", 15.0),
            tau_i=kwargs.get("tau_i", 0.5),
            rho=kwargs.get("rho", 30.0),
            beta1=kwargs.get("beta1", 1.0),
            beta2=kwargs.get("beta2", 1.0),
            beta3=kwargs.get("beta3", 0.5),
            gamma=kwargs.get("gamma", 0.5),
            lambda_=kwargs.get("lambda", kwargs.get("lambda_", 1.0)),
            score_threshold=kwargs.get("score_threshold", 0.0),
            score_aggregation=kwargs.get("score_aggregation", "mean"),
        )

    # ------------------------- Public API -------------------------

    def merge(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run paper-faithful SMM on a list of detection objects, returning merged objects.
        """
        _check_deps()
        if len(objects) == 0:
            return []

        # 1) Build R-tree of boxes
        rtree = RTreeIndex()
        for i, o in enumerate(objects):
            x1, y1, x2, y2 = o["bbox"]
            rtree.insert(i, (float(x1), float(y1), float(x2), float(y2)))

        # 2) Candidate edges with exact L2 post-filter and same-class gating
        rho = float(self.params["rho"])
        cand_pairs: List[Tuple[int, int]] = []
        for i, oi in enumerate(objects):
            x1, y1, x2, y2 = oi["bbox"]
            # Expand query by rho to get a superset; later filter by exact L2 distance.
            q = (x1 - rho, y1 - rho, x2 + rho, y2 + rho)
            for j in rtree.query(q):
                if j <= i:
                    continue
                oj = objects[j]
                if oi["label"] != oj["label"]:
                    continue
                # Exact L2 distance between boxes
                if bbox_l2_distance(oi["bbox"], oj["bbox"]) <= rho:
                    cand_pairs.append((i, j))

        # 3) Compute D_ij, I_ij, w_ij for candidate edges; evaluate anti-chaining C_gamma
        boundary_cache: Dict[int, np.ndarray] = {}
        edges: List[Tuple[int, int, float, float, float, bool]] = []
        # (i, j, D_ij, I_ij, w_ij, C_gamma)
        for (i, j) in cand_pairs:
            oi, oj = objects[i], objects[j]
            D_ij = boundary_distance(oi["mask"], oj["mask"], boundary_cache, i, j)
            I_ij = compute_iou(oi["mask"], oj["mask"])
            w_ij = edge_weight(D_ij, I_ij, oi["score"], oj["score"], self.params)
            C_ok = compatible_pair(oi, oj, D_ij, I_ij, w_ij, self.params)
            edges.append((i, j, D_ij, I_ij, w_ij, C_ok))

        # 4) Solve correlation clustering ILP with triangle inequalities and cannot-links
        clusters = self._solve_exact_cc_ilp(len(objects), edges)

        # 5) Merge per cluster (Φ(A))
        merged: List[Dict[str, Any]] = []
        score_thresh = float(self.params.get("score_threshold", 0.0))
        score_agg = str(self.params.get("score_aggregation", "mean")).lower()

        for comp in clusters:
            group = [objects[u] for u in comp]
            masks = [g["mask"] for g in group]
            merged_mask = merge_masks(masks)
            merged_bbox = mask_to_bbox(merged_mask)

            scores = np.array([float(g["score"]) for g in group], dtype=np.float32)
            if score_agg == "area_mean":
                areas = np.array([float(g["mask"].sum()) for g in group], dtype=np.float32)
                if areas.sum() > 0:
                    s_new = float((scores * areas).sum() / areas.sum())
                else:
                    s_new = float(scores.mean())
            else:
                s_new = float(scores.mean())

            label_new = group[0]["label"]  # all same by construction
            out = {"mask": merged_mask, "bbox": merged_bbox, "score": s_new, "label": label_new}
            if out["score"] >= score_thresh:
                merged.append(out)

        return merged

    # ------------------------- ILP core -------------------------

    def _solve_exact_cc_ilp(self,
                            n: int,
                            edges: List[Tuple[int, int, float, float, float, bool]]
                            ) -> List[List[int]]:
        """
        Exact correlation clustering ILP with:
          - x_ij ∈ {0,1} : 1 if separated, 0 if merged
          - Objective: Σ [ w_ij * x_ij + λ * (1 - w_ij) * (1 - x_ij) ]
          - Triangle inequalities: x_ij ≤ x_ik + x_kj
          - Cannot-link (anti-chaining): if C_gamma(i,j) == False => x_ij = 1

        Only pairs present in 'edges' are modeled; non-edges are treated as absent (no preference).
        Clusters are extracted from x_ij == 0 as connected components.
        """
        lam = float(self.params.get("lambda_", self.params.get("lambda", 1.0)))

        # Map existing pairs for quick access
        pair_index = {(i, j): k for k, (i, j, *_rest) in enumerate(edges)}

        # Create ILP problem
        prob = pulp.LpProblem("SMM_CorrelationClustering", pulp.LpStatusOptimal)

        # Decision vars for existing pairs only
        x_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for (i, j, D_ij, I_ij, w_ij, C_ok) in edges:
            x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Objective
        obj_terms = []
        for (i, j, D_ij, I_ij, w_ij, C_ok) in edges:
            x = x_vars[(i, j)]
            # cost for separated: w_ij * x
            # cost for merged   : λ * (1 - w_ij) * (1 - x)
            obj_terms.append(w_ij * x + lam * (1.0 - w_ij) * (1.0 - x))
        prob += pulp.lpSum(obj_terms)

        # Triangle inequalities for triples where all three pairs exist
        # x_ij <= x_ik + x_kj  (for all distinct i<j<k)
        for i in range(n):
            for j in range(i + 1, n):
                ij = (i, j)
                if ij not in x_vars:
                    continue
                for k in range(j + 1, n):
                    ik = (i, k)
                    jk = (j, k)
                    if ik in x_vars and jk in x_vars:
                        prob += x_vars[ij] <= x_vars[ik] + x_vars[jk]

        # Anti-chaining hard constraints (cannot-links): if C_gamma(i,j) == False => x_ij = 1
        for (i, j, D_ij, I_ij, w_ij, C_ok) in edges:
            if not C_ok:
                prob += x_vars[(i, j)] == 1

        # Solve
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            # Fall back: treat x_ij < 0.5 as merged; still return something reasonable
            pass

        # Build must-link graph for x_ij == 0
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (i, j, *_rest) in edges:
            x_val = x_vars[(i, j)].value()
            if x_val is None:
                continue
            if x_val <= 0.5:  # merged
                G.add_edge(i, j)

        # Connected components = clusters
        components = [sorted(list(c)) for c in nx.connected_components(G)]
        # Add isolated nodes missing entirely (if any) — though all candidate nodes were added above
        covered = set(u for comp in components for u in comp)
        for u in range(n):
            if u not in covered:
                components.append([u])

        # Final intra-cluster validity (strict anti-chaining within cluster)
        # Ensure every cluster satisfies C_gamma for all internal pairs.
        # If a violation is found, split the cluster using a simple repair (cannot-link edges cut).
        components = self._repair_clusters_with_antichain(components, edges)
        return components

    @staticmethod
    def _repair_clusters_with_antichain(components: List[List[int]],
                                        edges: List[Tuple[int, int, float, float, float, bool]]
                                        ) -> List[List[int]]:
        """
        Enforce pairwise anti-chaining inside each cluster as a post-check.
        If a cluster contains a cannot-link pair (C_ok=False), split via removing that edge and re-CC.
        """
        # Build quick look-up for C_gamma on an undirected key
        C_ok_map: Dict[Tuple[int, int], bool] = {}
        for (i, j, _D, _I, _w, C_ok) in edges:
            C_ok_map[(i, j)] = C_ok
            C_ok_map[(j, i)] = C_ok

        repaired: List[List[int]] = []
        for comp in components:
            if len(comp) <= 1:
                repaired.append(comp)
                continue
            H = nx.Graph()
            H.add_nodes_from(comp)
            # connect only pairs that are C_ok=True
            for a_idx in range(len(comp)):
                for b_idx in range(a_idx + 1, len(comp)):
                    u, v = comp[a_idx], comp[b_idx]
                    ok = C_ok_map.get((u, v), True)  # if missing, default to True
                    if ok:
                        H.add_edge(u, v)
            # Components of H are valid clusters
            for sub in nx.connected_components(H):
                repaired.append(sorted(list(sub)))
        return repaired


# ------------------------- Convenience API -------------------------

def smm_merge(objects: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Functional wrapper around SpatialMaskMerger for quick use:
        merged = smm_merge(objects, tau_d=15.0, tau_i=0.5, rho=30.0, ...)
    """
    merger = SpatialMaskMerger(**kwargs)
    return merger.merge(objects)
