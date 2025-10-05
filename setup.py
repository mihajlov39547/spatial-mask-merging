# setup.py
# Packaging configuration for the Spatial Mask Merging (SMM) project.
#
# Install (editable):   pip install -e .
# Source distribution:  python setup.py sdist bdist_wheel

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = ROOT / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else (
    "Spatial Mask Merging (SMM): paper-faithful implementation with exact correlation clustering."
)

setup(
    name="spatial-mask-merging",
    version="0.1.0",
    description="Spatial Mask Merging (SMM) — paper-faithful instance mask post-processing with exact correlation clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marko Mihajlović, with contributions from Marina Marjanović",
    author_email="",
    url="https://github.com/mihajlov39547/spatial-mask-merging",
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(exclude=("examples", "docs", "tests")),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "networkx>=3.0",
        "pulp>=2.7",
        "Pillow>=9.0.0",
        "rtree>=1.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/mihajlov39547/spatial-mask-merging",
        "Source": "https://github.com/mihajlov39547/spatial-mask-merging",
        "Tracker": "https://github.com/mihajlov39547/spatial-mask-merging/issues",
    },
    keywords=["computer-vision", "instance-segmentation", "post-processing", "mask-merging", "R-tree", "SMM"],
)
