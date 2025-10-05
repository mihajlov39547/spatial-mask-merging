# Spatial Mask Merging
**Marko Mihajlović**, with contributions from **Marina Marjanović**  
*Faculty of Informatics and Computing, Singidunum University, Belgrade, Serbia*

[![Release](https://img.shields.io/badge/release-v0.1.0--alpha-white.svg)](https://github.com/mihajlov39547/spatial-mask-merging/releases/tag/v0.1.0-alpha)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Paper DOI](https://img.shields.io/badge/DOI-10.3390/math13193079-red.svg)

Official implementation of the Spatial Mask Merging (SMM) algorithm, a post-processing algorithm designed to improve instance segmentation in high-resolution images. It addresses the limitations of traditional tiling methods by merging fragmented masks using graph clustering and spatial metrics.

---

## Highlights
- ⚡ Spatially optimized mask merging using R-tree indexing for efficient spatial queries  
- 🧩 Graph-based mask clustering for robust merging of overlapping and adjacent instances  
- 🧪 Pixel-level overlap and boundary distance metrics ensuring precise spatial consistency  
- 🔗 Anti-chaining constraint preventing indirect merges between dissimilar objects  
- 📦 Compatible with SAHI and other tiling-based inference pipelines for large-scale segmentation  
- 📈 Validated on the iSAID benchmark demonstrating significant precision and consistency gains  

---

## 🗂 Repository Structure

```
spatial-mask-merging/
│
├── docs/                         # Documentation and supporting materials
│   ├── algorithm_overview.md      # Algorithm description and math overview
│   ├── changelog.md               # Version history and updates
│   └── citation.bib               # Reference for academic citation
│
├── examples/                      # Example scripts and notebooks
│
├── smm/                           # Core SMM Python package
│   ├── __init__.py                # Package initialization
│   ├── predictions.py             # SMMPrediction data structure
│   ├── rtree_utils.py             # R-tree spatial indexing utilities
│   └── smm.py                     # Main SMM algorithm (ILP + Greedy)
│
├── tools/                        # Utilities (training, tuning, evaluation)
│   ├── optimize_smm.py           # Optuna-based hyperparameter optimizer for SMM
│   ├── visualization.py          # Visualize predictions or GT masks as PDFs for qualitative inspection
│   └── evaluation.py             # Batch evaluator (GPU-accelerated if PyTorch is available)
│
├── LICENSE                        # MIT License
├── README.md                      # Project readme (this file)
├── requirements.txt               # Python dependencies
└── setup.py                       # Package installation script
```

---

## Algorithm Overview

The **Spatial Mask Merging (SMM)** algorithm performs instance mask refinement by modeling predictions as nodes in a weighted graph, where edge weights express spatial and semantic consistency. The merging process is global, optimizing all candidate relations jointly instead of applying local, greedy rules.

### Main Components

1. **Graph Construction:**  
      Each predicted instance is treated as a vertex in a graph. Edges are established between masks that are spatially close, with edge weights reflecting three factors:  
   - *Spatial proximity* — normalized by distance threshold **τ_d**  
   - *Mask overlap* (IoU) — normalized by **τ_i**  
   - *Confidence consistency* — influenced by detection scores  

   The pairwise edge weight between instances *i* and *j* is computed as:

   ```
   w_ij = β₁ * (1 - D_ij / τ_d)_+ + β₂ * I_ij + β₃ * min(s_i, s_j)
   ```

   where  
   - **D_ij** – boundary distance between masks *M_i* and *M_j*  
   - **I_ij** – intersection-over-union (IoU) between masks  
   - **s_i**, **s_j** – detection confidences of the respective instances  

   The operator *(1 - D_ij / τ_d)_+* denotes the positive part of the normalized distance term (clamped at zero).  

2. **Spatial Pruning via R-tree:**  
   Neighboring masks are efficiently retrieved within a fixed search radius **ρ** using an R-tree spatial index. This ensures the algorithm remains scalable even with dense predictions.

3. **Global Optimization:**  
   Instance grouping is achieved using a correlation clustering objective that balances merging and separation penalties through the parameter **λ**. This provides globally consistent groupings rather than sequential local merges.

4. **Anti-Chaining Constraint:**  
   The anti-chaining threshold **γ** prevents indirect merging of incompatible objects, ensuring that all masks within a merged group are mutually compatible both geometrically and semantically.

---

## Algorithm Parameters

Spatial thresholds, edge weighting factors, and clustering penalties govern the behavior of the **Spatial Mask Merging (SMM)** algorithm. These parameters control the balance between over-merging and under-merging, spatial sensitivity, and candidate mask selection.

The parameters **τ_d**, **τ_i**, and **ρ** primarily regulate *candidate generation*, while the edge-weight coefficients **β₁**, **β₂**, **β₃**, the *correlation clustering penalty* **λ**, and the *anti-chaining threshold* **γ** influence *partitioning resolution*.  

Optimal values are **application-dependent** and vary based on object density, shape complexity, and whether inference is performed using tiling or full-image processing.

### Tunable Hyperparameters

| **Parameter** | **Description** | **Suggested Range** |
|----------------|-----------------|----------------------|
| **τ_d** | Distance scale (in pixels) used to normalize spatial proximity in edge weights *w_ij*. | 5–30 |
| **τ_i** | IoU threshold acting as a normalizing factor for overlap contribution. | 0.1–0.9 |
| **ρ** | R-tree search radius (in pixels) for candidate edge generation. | 10–50 |
| **β₁** | Weight of the distance contribution in *w_ij*. | 0.2–0.4 |
| **β₂** | Weight of the IoU contribution in *w_ij*. | 0.4–0.6 |
| **β₃** | Weight of the confidence contribution in *w_ij*. | 0.1–0.3 |
| **λ** | Correlation clustering penalty controlling over-merging vs. under-merging. | 0.1–2.0 |
| **γ** | Pairwise threshold enforcing the anti-chaining constraint. | 0.3–0.7 |

---

## Mask Merging Function Details

The merging function consolidates clustered detections into unified instances, ensuring semantic and spatial consistency across merged groups.

Given a valid merge group  
**A = {o₁, o₂, …, oₖ}**, where each object **oᵢ = (Mᵢ, bᵢ, sᵢ, ℓᵢ)** consists of a binary mask, bounding box, confidence score, and shared label, the merging function **Φ(A)** produces a new object **oₙₑw** as follows:

1. **Mask Fusion:**  
   Combine all masks using a pixel-wise logical OR.  
   This ensures that all covered pixels remain preserved in the merged result.

2. **Bounding Box Update:**  
   Compute the minimal axis-aligned rectangle enclosing all boxes in **A**.

3. **Score Aggregation:**  
   Compute the merged confidence either as an arithmetic mean or as a mask-size-weighted mean.

4. **Class Label Assignment:**  
   Since all members of **A** share the same class label, the merged object inherits this label.

The resulting merged object is  
**oₙₑw = (Mₙₑw, bₙₑw, sₙₑw, ℓₙₑw)**.

### Properties of Φ(A)
- **Idempotence:** Φ({o}) = o  
- **Symmetry:** Invariant to the ordering of elements in A  
- **Mask Preservation:** Ensures no loss of coverage, i.e., ⋃ₜ Mᵢₜ ⊆ Mₙₑw  

This formulation guarantees that merging consolidates spatially and semantically consistent detections into coherent instances, improving robustness of tiled inference without chained or inconsistent groupings.

---

## 📘 Documentation

For a detailed algorithmic explanation, see:  
[**algorithm_overview.md**](docs/algorithm_overview.md)

Changelog:  
[**changelog.md**](docs/changelog.md)

---

## 🧩 Installation

Clone this repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/mihajlov39547/spatial-mask-merging.git
cd spatial-mask-merging

# (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate # bash
or
.venv\Scripts\activate # windows

# Install required dependencies
pip install -r requirements.txt

---

## 🚀 Sample Usage

### 1) Core SMM (Python API)

```python
from smm.smm import SpatialMaskMerging
from smm.predictions import SMMPrediction

# Boolean numpy masks from your model
mask1 = ...
mask2 = ...
mask3 = ...

# Create prediction container
preds = [
    SMMPrediction(mask=mask1, score=0.91, label="car"),
    SMMPrediction(mask=mask2, score=0.88, label="car"),
    SMMPrediction(mask=mask3, score=0.82, label="car"),
]

# Choose backend: "ilp" (exact) or "greedy" (approximate)
smm = SpatialMaskMerging(mode="ilp", iou_weight=1.0, dist_weight=0.5, similarity_threshold=0.4)

merged = smm.merge(preds)
for obj in merged:
    print(obj.label, obj.score)
```

---

### 2) Hyperparameter Optimization (Optuna)

Run the Bayesian optimizer to tune SMM hyperparameters on a directory of prediction JSONs and matching ground-truth JSONs.

```bash
# From repo root
python tools/optimize_smm.py   --pred_dir /path/to/preds_json   --gt_dir /path/to/gt_json   --img_dir /path/to/images   --out_dir ./opt_results   --mode ilp   --trials 30
```

**Outputs (under `--out_dir`):**
- `best_params_ilp.json` — best hyperparameters found for ILP mode (filename includes mode).
- `smm_ilp_hparam_importance.json` and `.pdf` — parameter importances.
- `smm_ilp_optuna_trials.csv` — trials log with metrics and timings.

> Switch `--mode greedy` to optimize the greedy backend’s parameters instead.

---

### 3) Batch Evaluation

Evaluate a directory of *merged prediction JSONs* (e.g., the outputs after running SMM) against ground-truth:

```bash
python tools/evaluation.py   --pred_dir /path/to/merged_preds_json   --gt_dir /path/to/gt_json   --img_dir /path/to/images   --out_csv ./results/eval_ilp.csv   --iou_thr 0.5   --downscale 4
```

**Notes:**
- Uses GPU if PyTorch with CUDA is available; otherwise runs on CPU.
- `--downscale` reduces mask resolution for faster evaluation / lower VRAM.
- Metrics written to `--out_csv`: Precision, Recall, F1, Dice, PQ, Avg Fragments, Count Error, Mean Error (GPU path).

---

## 4) Visualization

Render prediction or ground-truth polygons onto source images and export as PDFs for visual inspection.

#### Example (predictions):
```bash
python tools/visualization.py --pred_dir /path/to/pred_jsons --image_dir /path/to/images
```

#### Example (ground-truth):
```bash
python tools/visualization.py --gt_dir /path/to/gt_labels --image_dir /path/to/images
```

**Notes:**
- Saves each visualization as `<image_name>_pred_visualization.pdf` or `_gt_visualization.pdf`.
- Supports both JSON (prediction format) and TXT (label format) inputs.
- Color map and class naming consistent with dataset definitions.
- Useful for checking polygon alignment and merging results after SMM.

---

### 5) Minimal Requirements

```bash
pip install -r requirements.txt
# Optional extras for speed/ILP:
pip install optuna torch pulp rtree opencv-python-headless matplotlib
```

---

## Citation
If you use this work, please cite:
```bibtex
@article{mihajlovic2025enhancing,
  title={Enhancing Instance Segmentation in High-Resolution Images Using Slicing-Aided Hyper Inference and Spatial Mask Merging Optimized via R-Tree Indexing},
  author={Mihajlovic, Marko and Marjanovic, Marina},
  journal={Mathematics},
  volume={13},
  number={19},
  pages={3079},
  year={2025},
  publisher={MDPI},
  note={(This article belongs to the Special Issue Mathematics Applications of Artificial Intelligence and Computer Vision)
}
```

## Copyright

Copyright (c) 2025 Marko Mihajlović, with contributions from Marina Marjanović

This repository provides the official implementation of the Spatial Mask Merging (SMM) algorithm.
All source code is distributed under the terms of the MIT License.
Users are permitted to use, modify, and distribute the software freely, provided that proper attribution is given through citation of the above publication.

For any use in academic, scientific, or derivative research, the citation of the original paper is strongly encouraged to acknowledge the theoretical and methodological contribution of this work.

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This research was conducted as part of the development of advanced post-processing techniques for high-resolution aerial imagery segmentation.

