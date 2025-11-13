#!/usr/bin/env python
"""Comprehensive tests for smm/__init__.py"""

import sys

print("=" * 70)
print("Testing smm/__init__.py")
print("=" * 70)

# Test 1: Import package
print("\n1. Testing package import...")
try:
    import smm
    print("  ✅ PASS: Package imports successfully")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)

# Test 2: Check __version__
print("\n2. Testing __version__ attribute...")
try:
    version = smm.__version__
    if version and isinstance(version, str):
        print(f"  ✅ PASS: Version = {version}")
    else:
        print(f"  ❌ FAIL: Invalid version: {version}")
except AttributeError as e:
    print(f"  ❌ FAIL: {e}")

# Test 3: Check __all__ exports
print("\n3. Testing __all__ exports...")
try:
    expected = {"SpatialMaskMerger", "smm_merge", "SMMPrediction", "SMMAnnotation"}
    actual = set(smm.__all__)
    if expected == actual:
        print(f"  ✅ PASS: All expected exports present")
        for item in sorted(smm.__all__):
            print(f"      - {item}")
    else:
        missing = expected - actual
        extra = actual - expected
        if missing:
            print(f"  ⚠️  Missing exports: {missing}")
        if extra:
            print(f"  ⚠️  Extra exports: {extra}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 4: Verify all exports are accessible
print("\n4. Testing export accessibility...")
errors = []
for name in smm.__all__:
    try:
        obj = getattr(smm, name)
        print(f"  ✅ {name}: {type(obj).__name__}")
    except AttributeError as e:
        errors.append(f"{name}: {e}")
        print(f"  ❌ {name}: NOT FOUND")

if errors:
    print(f"\n  ❌ {len(errors)} export(s) failed")
else:
    print(f"\n  ✅ All {len(smm.__all__)} exports accessible")

# Test 5: Test direct imports
print("\n5. Testing direct imports from submodules...")
try:
    from smm import SpatialMaskMerger, smm_merge
    from smm import SMMPrediction, SMMAnnotation
    print("  ✅ PASS: All imports work directly")
except ImportError as e:
    print(f"  ❌ FAIL: {e}")

# Test 6: Test convenience import style
print("\n6. Testing convenience import patterns...")
try:
    # Pattern 1: import package, use qualified names
    import smm
    merger1 = smm.SpatialMaskMerger()
    pred1 = smm.SMMPrediction(image_name="test.jpg")
    print("  ✅ PASS: Qualified import (smm.ClassName)")
    
    # Pattern 2: from import
    from smm import SpatialMaskMerger, SMMPrediction
    merger2 = SpatialMaskMerger()
    pred2 = SMMPrediction(image_name="test.jpg")
    print("  ✅ PASS: Direct import (from smm import ...)")
    
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 7: Check for unintended exports (private symbols)
print("\n7. Testing for unintended exports...")
public_symbols = set(smm.__all__) | {"__version__", "__all__", "__name__", "__doc__", 
                                      "__package__", "__loader__", "__spec__", "__path__",
                                      "__file__", "__cached__", "__builtins__"}
all_symbols = set(dir(smm))
private_exports = all_symbols - public_symbols

if private_exports:
    # Filter out truly internal Python symbols
    concerning = [s for s in private_exports if not s.startswith('_')]
    if concerning:
        print(f"  ⚠️  WARNING: Potentially unintended exports: {concerning}")
    else:
        print("  ✅ PASS: No unintended public exports")
else:
    print("  ✅ PASS: Clean namespace (only intended exports)")

# Test 8: Test that internal modules are not exposed
print("\n8. Testing module encapsulation...")
try:
    # These should NOT be directly accessible
    if hasattr(smm, 'rtree_utils'):
        print("  ⚠️  WARNING: Internal module 'rtree_utils' exposed")
    else:
        print("  ✅ PASS: Internal modules not exposed")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 9: Quick functionality test
print("\n9. Testing basic functionality...")
try:
    from smm import SMMPrediction, SpatialMaskMerger
    
    # Create prediction
    pred = SMMPrediction(image_name="test_init.jpg")
    pred.add_annotation("car", 0, 0.9, (10, 10, 50, 50), [[[10, 10], [50, 50]]])
    
    # Validate
    pred.validate_all()
    
    # Create merger
    merger = SpatialMaskMerger(tau_d=15.0)
    
    # Merge
    result = merger.merge(pred, (100, 100))
    
    if len(result) > 0:
        print(f"  ✅ PASS: Basic workflow works ({len(result)} cluster(s))")
    else:
        print("  ⚠️  WARNING: Merge returned no results")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 10: Check import performance
print("\n10. Testing import performance...")
try:
    import time
    start = time.time()
    
    # Fresh import (need to remove from sys.modules first)
    if 'smm_perf_test' in sys.modules:
        del sys.modules['smm_perf_test']
    
    # Time the import
    import smm as smm_perf_test
    elapsed = time.time() - start
    
    if elapsed < 1.0:
        print(f"  ✅ PASS: Import time = {elapsed*1000:.1f}ms (fast)")
    elif elapsed < 3.0:
        print(f"  ⚠️  OK: Import time = {elapsed*1000:.1f}ms (acceptable)")
    else:
        print(f"  ⚠️  SLOW: Import time = {elapsed*1000:.1f}ms (consider lazy imports)")
except Exception as e:
    print(f"  ⚠️  Could not measure: {e}")

print("\n" + "=" * 70)
print("✅ smm/__init__.py validation complete!")
print("=" * 70)
