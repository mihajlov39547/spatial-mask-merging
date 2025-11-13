# Release Notes - Spatial Mask Merging v1.0.1

**Release Date:** November 13, 2025  
**Type:** Minor Update - Code Refactoring & Maintenance  
**Status:** Production Ready ✅

---

## 🎯 Overview

Version 1.0.1 focuses on code quality improvements through comprehensive refactoring of the tools module. This release eliminates significant code duplication (~500 lines) by extracting shared GPU evaluation logic into a dedicated module, following DRY (Don't Repeat Yourself) principles.

**Key Improvement:** Single source of truth for GPU evaluation logic across all tools.

---

## 🆕 What's New

### New Module: `tools/gpu_evaluation.py` (403 lines)

Centralized GPU evaluation utilities now shared across all tools:

**Core Functions:**
- `compute_metrics_gpu()` - GPU-accelerated evaluation with adaptive chunking
- `compute_metrics_cpu()` - CPU fallback evaluator
- `load_mask_from_segmentation()` - Polygon to mask conversion with downscaling
- `ensure_class_ids()` - Class ID inference from annotations
- `_pairwise_iou_masks_torch()` - Vectorized mask IoU via GPU matmul
- `_boxes_from_stack_stable()` - Bounding box extraction on GPU
- `_auto_pred_chunk()` - Adaptive VRAM management (60% budget formula)

**Constants:**
- `USE_CUDA`, `DEVICE` - GPU configuration
- `CHUNK_P=1024`, `CHUNK_G=256` - Chunking parameters
- `ADAPTIVE_P=True` - Adaptive chunking enabled
- `IOU_THRESHOLD=0.5` - Matching threshold
- `DOWNSCALE_FACTOR=4` - Mask downscaling factor

### New Test Suite: `tests/test_refactored_tools.py`

Comprehensive validation of refactored architecture:
- ✅ Module import verification
- ✅ Function availability checks
- ✅ Shared module integration validation
- ✅ GPU/CPU evaluator functional tests
- ✅ Result consistency verification

**Test Coverage:** 7 test stages, all passing

### Helper Scripts

- `run_tests.bat` (Windows) - Easy test execution
- `run_tests.sh` (Linux/Mac) - Cross-platform support

---

## 🔧 Refactored Components

### `tools/evaluation.py` (207 lines, -55% reduction)

**Changes:**
- Removed ~250 lines of duplicate GPU evaluation code
- Now imports from `gpu_evaluation` module
- Maintained unique functions: `read_json()`, `get_image_shape()`, `evaluate_dir()`

**Benefits:**
- Cleaner, more maintainable code
- Consistent evaluation logic with other tools
- Faster bug fixes (single point of change)

### `tools/optimize_smm.py` (361 lines, -41% reduction)

**Changes:**
- Removed ~250 lines of duplicate GPU evaluation code
- Now imports from `gpu_evaluation` module
- Maintained unique functions: `run_smm_on_entry()`, `mask_to_polygons()`, `suggest_params()`, `make_objective()`

**Benefits:**
- Simplified codebase structure
- Identical evaluation metrics with `evaluation.py`
- Easier testing and debugging

---

## 📊 Code Quality Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total tool lines** | ~968 | ~971* | -500 duplicates |
| **evaluation.py** | ~457 | 207 | -55% |
| **optimize_smm.py** | ~611 | 361 | -41% |
| **Duplicate code** | ~500 lines | 0 lines | -100% |
| **Compilation errors** | 0 | 0 | ✅ Maintained |
| **Test coverage** | Manual | Automated | ✅ Improved |

*Includes new 403-line shared module, net reduction of ~500 duplicate lines

---

## ✨ Benefits

### For Developers
- **Single Source of Truth:** GPU evaluation logic centralized in one module
- **Easier Maintenance:** Bug fixes and optimizations apply once, propagate everywhere
- **Consistent Behavior:** Identical evaluation logic across all tools
- **Better Testing:** Centralized test coverage for GPU functions
- **Cleaner Code:** Reduced complexity and improved readability

### For Users
- **Same Functionality:** No breaking changes, all tools work identically
- **Better Reliability:** Centralized code = fewer bugs
- **Faster Updates:** Improvements to GPU evaluation benefit all tools simultaneously
- **Consistent Results:** Identical metrics across evaluation and optimization workflows

---

## 🔄 Migration Guide

### No Action Required ✅

This release is **100% backward compatible**. All tools maintain the same:
- Command-line interfaces
- Input/output formats
- Behavior and results
- Dependencies

Your existing scripts and workflows will continue to work without modification.

### Updated Usage (Same as v1.0.0)

**Batch Evaluation:**
```bash
python tools/evaluation.py \
  --pred_dir /path/to/preds \
  --gt_dir /path/to/gt \
  --img_dir /path/to/images \
  --out_csv results.csv
```

**Hyperparameter Optimization:**
```bash
python tools/optimize_smm.py \
  --data_dir /path/to/data \
  --study_name my_study \
  --n_trials 30
```

**Visualization:**
```bash
python tools/visualization.py preds \
  --pred-base-dir /path/to/preds \
  --image-dir /path/to/images
```

---

## 🧪 Testing & Validation

### Verification Performed

✅ **Compilation:** All scripts compile without errors  
✅ **Imports:** All modules import successfully  
✅ **Integration:** Shared module correctly used by both tools  
✅ **Functionality:** GPU/CPU evaluators work with sample data  
✅ **Consistency:** Identical results across GPU/CPU paths  
✅ **Regression:** No breaking changes detected

### Run Tests Yourself

```bash
# Windows
.\run_tests.bat

# Linux/Mac
./run_tests.sh

# Or with activated venv
python tests/test_refactored_tools.py
```

---

## 📦 Installation

### Upgrade from v1.0.0

```bash
cd spatial-mask-merging
git pull origin main
# No dependency changes, environment already compatible
```

### Fresh Installation

```bash
git clone https://github.com/mihajlov39547/spatial-mask-merging.git
cd spatial-mask-merging
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python check_env.py
```

---

## 📚 Documentation Updates

### Updated Files
- `README.md` - Added `gpu_evaluation.py` to repository structure
- `README.md` - Updated architecture notes with DRY principles
- `README.md` - Added shared module performance characteristics

### New Files
- `tests/test_refactored_tools.py` - Comprehensive refactoring validation
- `run_tests.bat` / `run_tests.sh` - Test execution helpers

---

## 🐛 Bug Fixes

None. This release focuses purely on code quality improvements through refactoring.

---

## ⚠️ Breaking Changes

**None.** This is a fully backward-compatible release.

---

## 🔮 Future Enhancements

Potential improvements for future releases:
- Unit tests for individual `gpu_evaluation.py` functions
- Configuration file for GPU constants (CHUNK_P, CHUNK_G, etc.)
- Performance profiling and benchmarking suite
- Additional shared utilities extraction

---

## 📊 Performance Impact

**No performance changes.** This release maintains identical:
- Runtime characteristics
- Memory usage patterns
- GPU/CPU behavior
- Evaluation accuracy

The refactoring is purely structural, with no algorithmic changes.

---

## 🤝 Contributing

The refactored architecture makes contributions easier:
- **Add evaluation metrics:** Update `gpu_evaluation.py` once
- **Fix GPU bugs:** Single file to modify
- **Optimize performance:** Benefits all tools immediately
- **Write tests:** Centralized testing for shared code

---

## 📖 Version History

### v1.0.1 (November 13, 2025) - Code Refactoring
- ✅ Created shared `gpu_evaluation.py` module
- ✅ Refactored `evaluation.py` (-55% lines)
- ✅ Refactored `optimize_smm.py` (-41% lines)
- ✅ Eliminated ~500 lines of duplicate code
- ✅ Added comprehensive test suite
- ✅ Updated documentation

### v1.0.0 (November 2025) - Initial Release
- ✅ Core SMM algorithm with ILP solver
- ✅ GPU-accelerated evaluation tools
- ✅ Optuna-based hyperparameter optimization
- ✅ Comprehensive documentation
- ✅ Production-ready code quality

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/mihajlov39547/spatial-mask-merging/issues)
- **Documentation:** [README.md](README.md), [docs/algorithm_overview.md](docs/algorithm_overview.md)
- **Citation:** See [CITATION.cff](CITATION.cff) or [docs/citation.bib](docs/citation.bib)

---

## 🎓 Citation

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
  publisher={MDPI}
}
```

---

**Thank you for using Spatial Mask Merging!**

This release continues our commitment to high-quality, maintainable, production-ready code for advanced instance segmentation in high-resolution images.
