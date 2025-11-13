# Algorithm Overview — Spatial Mask Merging (SMM)

**Author:** Marko Mihajlović, with contributions from Marina Marjanović
**License:** MIT  
**Version:** 1.0.1  
**Date:** November 13, 2025  

---

## 1. Introduction

**Spatial Mask Merging (SMM)** is a post-processing algorithm designed to fuse overlapping instance masks into coherent object hypotheses.  
It is **paper-faithful** to the method described in *"Enhancing Instance Segmentation in High-Resolution Images Using Slicing-Aided Hyper Inference and Spatial Mask Merging Optimized via R-Tree Indexing"* (Mathematics, MDPI 2025), implementing both the **exact correlation clustering formulation** and its **efficient greedy approximation**.

The algorithm integrates spatial, semantic, and structural cues to ensure consistent merging of overlapping predictions from segmentation models.

---

## 2. Problem Definition

Given a set of predicted binary masks  
***M = {M₁, M₂, …, Mₙ}***,  
each with confidence ***sᵢ*** and optional class label ***cᵢ***,  
the goal is to merge spatially and semantically redundant masks into disjoint object instances.

### Objective

We seek an assignment that maximizes intra-cluster similarity while minimizing overlap across distinct clusters.

Formally, SMM solves:  
***minₓ Σᵢ<ⱼ wᵢⱼ (1 - xᵢⱼ)***  
subject to transitivity constraints on the clustering variables ***xᵢⱼ***,  
where ***wᵢⱼ*** is a learned or computed similarity score.

This is equivalent to the **correlation clustering** problem.

---

## 3. Core Components

### 3.1 Similarity Computation

For each pair of masks ***Mᵢ, Mⱼ***, we compute:

- **Spatial proximity:** Boundary distance ***Dᵢⱼ*** normalized by threshold ***τ_d***  
- **Mask overlap (IoU):** Intersection-over-Union ***Iᵢⱼ*** normalized by ***τ_i***  
- **Confidence consistency:** Detection scores ***sᵢ, sⱼ***

The edge weight ***wᵢⱼ*** is computed as:  
***wᵢⱼ = β₁ · (1 - Dᵢⱼ/τ_d)₊ + β₂ · Iᵢⱼ + β₃ · min(sᵢ, sⱼ)***  
where ***(·)₊*** denotes the positive part (max(0, ·)), and ***β₁, β₂, β₃ > 0*** are weight coefficients (typically summing to 1.0).

---

### 3.2 Graph Construction

Masks form the vertices of a weighted graph ***G = (V, E)***:  
- Each node ***vᵢ ∈ V*** represents a mask.  
- Each edge ***(i, j) ∈ E*** connects spatially adjacent masks (within search radius ***ρ***).

Edges are discovered using an **R-tree spatial index**, enabling efficient neighborhood queries with O(log N) complexity. **Triangle inequality constraints** are applied during graph construction to reduce edge count from O(N³) to O(E·N) for sparse graphs, yielding 50-100% speedup.

---

### 3.3 Optimization: Exact ILP Solver

When high accuracy is required, SMM uses **Integer Linear Programming (ILP)** to find the global optimum of the correlation clustering problem.

We solve:  
***minₓ Σ₍ᵢ,ⱼ₎ wᵢⱼ (1 - xᵢⱼ) + λ · Σ₍ᵢ,ⱼ₎ xᵢⱼ***  
subject to:  
- ***xᵢⱼ + xⱼₖ - xᵢₖ ≤ 1*** (transitivity)  
- ***xᵢⱼ = 0*** if ***wᵢⱼ < γ*** (anti-chaining constraint)

where ***λ*** is the clustering penalty (merge vs. separate trade-off) and ***γ*** is the pairwise compatibility threshold.  
The implementation uses the `pulp` library as a generic solver interface (default: CBC), with optional support for commercial solvers (Gurobi, CPLEX).

---

### 3.4 Greedy Approximation (Baseline)

A faster, deterministic baseline is provided for large-scale datasets.  
It iteratively merges the most similar mask pairs until no mergeable pairs remain (based on a similarity threshold).

This version scales linearly with the number of edges.

---

## 4. Implementation Details

### Dependencies
**Core:**
- `numpy`, `scipy`, `networkx` — numerical and graph operations  
- `pulp` — ILP solver interface  
- `rtree` — R-tree spatial indexing (with pure-Python fallback)

**Tools (v1.0.1 - Refactored):**
- `torch` (optional) — GPU-accelerated evaluation via `tools/gpu_evaluation.py`  
- `optuna` — hyperparameter optimization  
- `opencv-python` (cv2) — mask processing  
- `pandas`, `tqdm` — data handling and progress tracking

### Key Classes
- `SpatialMaskMerger`: main algorithm class (mode="ilp" or "greedy")  
- `SMMPrediction`: standardized input container for predicted masks and metadata  
- `SMMAnnotation`: annotation data structure with validation  
- `RTreeIndex`: spatial search utility for bounding box queries (auto-optimized)

### Key Files
- Core implementation: `smm/smm.py` (ILP + Greedy solvers)  
- Data structures: `smm/predictions.py`  
- Spatial indexing: `smm/rtree_utils.py`  
- Shared GPU evaluation: `tools/gpu_evaluation.py` (v1.0.1)

---

## 5. Output

SMM outputs:
- A set of **merged masks** representing disjoint object instances.  
- **Cluster assignments** for each original prediction.  
- Optional **merge graphs** or **confidence maps** for analysis.

---

## 6. Complexity and Performance

| Variant | Time Complexity | Optimality | Typical Use Case |
|----------|-----------------|-------------|------------------|
| ILP-based (optimized) | ***O(E·N)*** for sparse graphs, ***O(N³)*** worst-case | Exact | Production evaluation, medium datasets (<500 masks) |
| Greedy | ***O(E log N)*** with R-tree | Approximate | Real-time inference, large-scale datasets (500+ masks) |

**Performance Optimizations (v1.0.0 - v1.0.1):**
- Triangle inequality filtering: 50-100% speedup  
- Vectorized IoU computation: 2-3× faster  
- R-tree spatial queries: O(log N) vs O(N) brute force  
- Shared GPU evaluation module: Consistent metrics, easier maintenance (v1.0.1)

---

## 7. References

- **Mihajlović, M.** and **Marjanović, M.** (2025). *Enhancing Instance Segmentation in High-Resolution Images Using Slicing-Aided Hyper Inference and Spatial Mask Merging Optimized via R-Tree Indexing*. Mathematics, 13(19), 3079. MDPI. DOI: [10.3390/math13193079](https://doi.org/10.3390/math13193079)  
- Bansal, N., Blum, A., Chawla, S. (2004). *Correlation Clustering*. Machine Learning, 56, 89-113.

---

## 8. License

This work is licensed under the **MIT License**.  
See `LICENSE` for full terms.
