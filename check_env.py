#!/usr/bin/env python
"""
Quick environment checker for Spatial Mask Merging project.
Run this to verify all dependencies are installed correctly.
"""

import sys

def check_package(name, import_name=None):
    """Try to import a package and report status."""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✅ {name}")
        return True
    except ImportError as e:
        print(f"❌ {name} - {e}")
        return False

def main():
    print("=" * 60)
    print("Spatial Mask Merging - Environment Check")
    print("=" * 60)
    print(f"\nPython: {sys.version}\n")
    
    print("Core Dependencies:")
    print("-" * 60)
    core_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("networkx", "networkx"),
        ("pulp", "pulp"),
        ("rtree", "rtree"),
        ("Pillow", "PIL"),
    ]
    
    core_ok = all(check_package(name, imp) for name, imp in core_packages)
    
    print("\nTools Dependencies:")
    print("-" * 60)
    tools_packages = [
        ("opencv-python-headless", "cv2"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("optuna", "optuna"),
        ("matplotlib", "matplotlib"),
    ]
    
    tools_ok = all(check_package(name, imp) for name, imp in tools_packages)
    
    print("\nOptional Dependencies:")
    print("-" * 60)
    optional_packages = [
        ("torch", "torch"),
    ]
    
    for name, imp in optional_packages:
        check_package(name, imp)
    
    print("\nProject Modules:")
    print("-" * 60)
    project_ok = True
    try:
        from smm.smm import SpatialMaskMerger
        from smm.predictions import SMMPrediction, SMMAnnotation
        from smm.rtree_utils import RTreeIndex
        print("✅ smm.smm (SpatialMaskMerger)")
        print("✅ smm.predictions (SMMPrediction, SMMAnnotation)")
        print("✅ smm.rtree_utils (RTreeIndex)")
    except ImportError as e:
        print(f"❌ SMM package - {e}")
        project_ok = False
    
    print("\n" + "=" * 60)
    if core_ok and tools_ok and project_ok:
        print("✅ ALL CHECKS PASSED - Environment is ready!")
    else:
        print("⚠️  Some packages are missing. Run:")
        print("   pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
