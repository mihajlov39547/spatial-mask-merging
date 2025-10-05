# Algorithm Overview — Spatial Mask Merging (SMM)

**Author:** Marko Mihajlović (and collaborators)  
**License:** MIT  
**Version:** 1.1.0  
**Date:** 2025-10-05  

---

## 1. Introduction

**Spatial Mask Merging (SMM)** is a post-processing algorithm designed to fuse overlapping instance masks into coherent object hypotheses.  
It is **paper-faithful** to the method described in *"Spatial Mask Merging for Accurate Instance Segmentation"*, implementing both the **exact correlation clustering formulation** and its **efficient greedy approximation**.

The algorithm integrates spatial, semantic, and structural cues to ensure consistent merging of overlapping predictions from segmentation models.

---

## 2. Problem Definition

Given a set of predicted binary masks  
\[
\mathcal{M} = \{M_1, M_2, \dots, M_N\},
\]  
each with confidence \(s_i\) and optional class label \(c_i\),  
the goal is to merge spatially and semantically redundant masks into disjoint object instances.

### Objective

We seek an assignment that maximizes intra-cluster similarity while minimizing overlap across distinct clusters.

Formally, SMM solves:

\[
\min_{x_{ij}} \sum_{i<j} w_{ij} (1 - x_{ij}),
\]
subject to transitivity constraints on the clustering variables \(x_{ij}\),  
where \(w_{ij}\) is a learned or computed similarity score.

This is equivalent to the **correlation clustering** problem.

---

## 3. Core Components

### 3.1 Similarity Computation

For each pair of masks \(M_i, M_j\), we compute:

- **Spatial IoU:** Intersection-over-Union of mask regions.  
- **Centroid distance:** Euclidean distance between mask centroids.  
- **Semantic similarity:** Cosine similarity of class probability vectors (if available).

A combined score \(w_{ij}\) is derived as:
\[
w_{ij} = \alpha \cdot \text{IoU}(M_i, M_j) - \beta \cdot \text{dist}(M_i, M_j)
\]
with tunable hyperparameters \(\alpha, \beta > 0\).

---

### 3.2 Graph Construction

Masks form the vertices of a weighted graph \(G = (V, E)\):  
- Each node \(v_i \in V\) represents a mask.  
- Each edge \((i, j) \in E\) connects masks with nonzero overlap or proximity.

Edges are discovered using an **R-tree spatial index**, enabling efficient neighborhood queries.

---

### 3.3 Optimization: Exact ILP Solver

When high accuracy is required, SMM uses **Integer Linear Programming (ILP)** to find the global optimum of the correlation clustering problem.

We solve:
\[
\min_{x_{ij} \in \{0,1\}} \sum_{(i,j)} c_{ij} x_{ij}
\]
subject to:
\[
x_{ij} + x_{jk} - x_{ik} \le 1, \quad \forall i,j,k
\]

This ensures a consistent clustering.  
The implementation uses the `pulp` library as a generic solver interface, allowing backends such as CBC or Gurobi.

---

### 3.4 Greedy Approximation (Baseline)

A faster, deterministic baseline is provided for large-scale datasets.  
It iteratively merges the most similar mask pairs until no mergeable pairs remain (based on a similarity threshold).

This version scales linearly with the number of edges.

---

## 4. Implementation Details

### Dependencies
- `numpy`, `scipy`, `networkx` — core numerical and graph operations  
- `pulp` — ILP solver interface (optional)  
- `rtree` or `pygeos.STRtree` — spatial index acceleration (optional)

### Key Classes
- `SMMPrediction`: standardized input container for predicted masks and metadata.
- `RTreeIndex`: optional spatial search utility for bounding box queries.

### File
Core implementation: `smm/smm.py`

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
| ILP-based | \(O(N^3)\) | Exact | Benchmark evaluation, small datasets |
| Greedy | \(O(N \log N)\) | Approximate | Real-time or large-scale inference |

---

## 7. References

- Mihajlović, M. *et al.*, **Spatial Mask Merging for Accurate Instance Segmentation**, 2025.  
- Bansal, N., Blum, A., Chawla, S. *Correlation Clustering*. Machine Learning, 2004.

---

## 8. License

This work is licensed under the **MIT License**.  
See `LICENSE` for full terms.

