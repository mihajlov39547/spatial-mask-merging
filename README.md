# Spatial Mask Merging

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Paper DOI](https://img.shields.io/badge/DOI-10.3390/math13193079-red.svg)

Official implementation of the Spatial Mask Merging (SMM) algorithm, a post-processing algorithm designed to improve instance segmentation in high-resolution images. It addresses the limitations of traditional tiling methods by merging fragmented masks using graph clustering and spatial metrics.

---

## Highlights
- ⚡ Spatially optimized mask merging using R-tree indexing for efficient spatial queries
- 🧩 Graph-based mask clustering for robust merging of overlapping and adjacent instances
- 🧪 Pixel-level overlap and boundary distance metrics ensuring precise spatial consistency
- 📦 Compatible with SAHI and other tiling-based inference pipelines for large-scale segmentation
- 📈 Validated on the iSAID benchmark demonstrating significant precision and consistency gains

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
  publisher={MDPI}
}
```

## Copyright

Copyright (c) 2025 Marko Mihajlović

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

