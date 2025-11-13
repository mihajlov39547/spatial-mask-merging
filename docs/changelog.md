# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Planned
- **Configuration file for GPU constants:** YAML/JSON config for CHUNK_P, CHUNK_G, IOU_THRESHOLD, DOWNSCALE_FACTOR (allows users to tune GPU memory usage without code changes)
- **Extended benchmark suite:** Automated evaluation on public datasets (iSAID, COCO, Cityscapes) with standardized metrics
- **Paper reproduction scripts:** Complete pipeline to reproduce paper figures, tables, and experimental results with documented hyperparameters
- **Performance profiling tools:** Benchmarking suite with detailed timing breakdowns and memory usage analysis
- **Enhanced visualization:** Interactive HTML reports for evaluation results with per-image metrics and error analysis

---

## [1.0.1] - 2025-11-13
### Added
- **Shared GPU Evaluation Module:** Created `tools/gpu_evaluation.py` (403 lines) centralizing GPU evaluation utilities
  - `compute_metrics_gpu()` - GPU-accelerated evaluation with adaptive chunking
  - `compute_metrics_cpu()` - CPU fallback evaluator
  - `load_mask_from_segmentation()` - Polygon to mask conversion with downscaling
  - `ensure_class_ids()` - Class ID inference from annotations
  - GPU helper functions: `_pairwise_iou_masks_torch()`, `_boxes_from_stack_stable()`, `_auto_pred_chunk()`
  - Constants: USE_CUDA, DEVICE, CHUNK_P, CHUNK_G, ADAPTIVE_P, IOU_THRESHOLD, DOWNSCALE_FACTOR
- **Test Suite:** `tests/test_refactored_tools.py` with 7 validation stages for refactored architecture
- **Helper Scripts:** `run_tests.bat` and `run_tests.sh` for easy test execution
- **Release Notes:** Comprehensive `RELEASE_NOTES_v1.0.1.md` documenting refactoring improvements

### Changed
- **Refactored `tools/evaluation.py`:** Reduced from ~457 to 207 lines (-55%) by importing from `gpu_evaluation` module
  - Removed ~250 lines of duplicate GPU evaluation code
  - Maintained unique functions: `read_json()`, `get_image_shape()`, `evaluate_dir()`
- **Refactored `tools/optimize_smm.py`:** Reduced from ~611 to 361 lines (-41%) by importing from `gpu_evaluation` module
  - Removed ~250 lines of duplicate GPU evaluation code
  - Maintained unique functions: `run_smm_on_entry()`, `mask_to_polygons()`, `suggest_params()`, `make_objective()`
- **Updated Documentation:**
  - `README.md` - Added `gpu_evaluation.py` to repository structure, DRY architecture notes
  - `index.html` - Updated to v1.0.1 with refactoring highlights and professional styling
  - `docs/algorithm_overview.md` - Updated version, formulas, dependencies, and performance metrics

### Fixed
- Eliminated ~500 lines of duplicate GPU evaluation code across tools
- Centralized GPU evaluation logic for consistent behavior and easier maintenance

### Performance
- No runtime performance changes (purely structural refactoring)
- Maintained identical behavior, metrics, and memory usage

---

## [1.0.0] - 2025-11-13
### Added
- **Production-Ready Core:** Enterprise-grade SMM algorithm with comprehensive optimization
  - Triangle inequality filtering: O(N³) → O(E·N) for sparse graphs (50-100% speedup)
  - Vectorized IoU computation: 2-3× faster with NumPy broadcasting
  - Batch edge weight calculation: 100-200% performance gain
  - Optimized R-tree queries: 5-10% improvement
- **Comprehensive Testing:** 35+ unit tests covering functionality, edge cases, and integration
  - `tests/test_smm_optimization.py` - Core algorithm performance validation
  - `tests/test_smm_validation.py` - Input validation and error handling
  - `tests/test_predictions_validation.py` - Data structure validation (10 tests)
  - `tests/test_predictions_new_features.py` - New features validation (5 tests)
  - `tests/test_rtree_validation.py` - R-tree query validation
  - `tests/test_rtree_new_features.py` - Introspection and optimization detection
  - `tests/test_init_validation.py` - API namespace and version checks (10 tests)
  - `tests/verify_readme_claims.py` - Production readiness verification (8 checks)
- **GPU-Accelerated Tools:**
  - `tools/evaluation.py` - CUDA-enabled batch evaluation with automatic CPU fallback
  - `tools/optimize_smm.py` - Optuna-based hyperparameter optimization with GPU evaluation
  - `tools/visualization.py` - PDF rendering with polygon overlays
- **Data Structure Enhancements:**
  - `validate_all()` method for batch validation in `SMMPrediction`
  - Optimized `bbox_to_mask()` rasterization (10-15% faster)
  - Enhanced validation with detailed error messages
- **API Improvements:**
  - Clean namespace with zero internal module pollution
  - 100% type hint coverage for IDE support
  - Comprehensive input validation with actionable error messages
- **Documentation:**
  - Production-ready README with comprehensive examples
  - Professional `index.html` website with enterprise-grade presentation
  - `RELEASE_NOTES_v1.0.0.md` with detailed optimization metrics

### Changed
- **Clean Public API:** Removed internal module pollution from `smm.__init__.py`
- **R-tree Utilities:** Added `is_optimized()` introspection method
- **Version Bump:** Updated to 1.0.0 reflecting production-ready status

### Performance Improvements
| Component | Optimization | Speedup |
|-----------|-------------|---------|
| Graph Construction | Triangle inequality filtering | 50-100% |
| IoU Computation | Vectorized operations | 2-3× |
| Edge Weights | Batch NumPy broadcasting | 100-200% |
| R-tree Queries | Optimized bbox checks | 5-10% |
| Mask Operations | Optimized rasterization | 10-15% |

---

## [0.1.0-alpha] - 2025-09-20
### Added
- Initial alpha release of the **Spatial Mask Merging (SMM)** framework
- Implemented baseline greedy merging algorithm
- Added reproducible test utilities and example usage
- Defined project structure under `smm/` with core module
- Exact correlation clustering (ILP) implementation using `pulp` solver
- R-tree spatial indexing wrapper for efficient queries
- `SMMPrediction` standardized container for inputs
- Full docstrings and paper-faithful algorithm implementation

### Known Limitations
- Limited optimization of core algorithm
- Basic validation and error handling
- Minimal test coverage
- No GPU acceleration support

---

