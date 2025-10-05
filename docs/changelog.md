# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Planned
- Integration of GPU-accelerated ILP solvers.
- Optional PyTorch interface for end-to-end differentiable merging.
- Extended benchmark suite with public datasets.
- Paper reproduction scripts and metrics report generator.

---

## [1.1.0] - 2025-10-05
### Added
- **Exact correlation clustering (ILP)** implementation using the `pulp` solver.
- Added `RTreeIndex` wrapper for efficient spatial queries during merging.
- Introduced the `SMMPrediction` standardized container for public inputs.
- Added full docstring and in-code comments describing algorithmic intent and paper fidelity.

### Changed
- Refactored merging pipeline to strictly follow the *paper-faithful Spatial Mask Merging* formulation.
- Improved handling of spatial masks using `scipy.ndimage.binary_erosion`.
- Modularized dependency management (`pulp` and `rtree` now optional).

### Fixed
- Corrected distance calculations with `scipy.spatial.distance.cdist`.
- Fixed potential graph disconnection issues in `networkx` clustering phase.

---

## [1.0.0] - 2025-09-20
### Added
- Initial release of the **Spatial Mask Merging (SMM)** framework.
- Implemented baseline greedy merging algorithm.
- Added reproducible test utilities and example usage.
- Defined project structure under `smm/` with core module and placeholders for utilities.

---

