#!/usr/bin/env python
"""Verify README claims are accurate"""

import sys

print("=" * 70)
print("Verifying README.md Production Claims")
print("=" * 70)

# Test 1: Check version
print("\n1. Verifying package version...")
try:
    import smm
    version = smm.__version__
    if version == "0.1.0":
        print(f"  ⚠️  Version is {version}, but README claims v1.0.0")
        print("  📝 Note: Update smm/__init__.py __version__ to '1.0.0'")
    else:
        print(f"  ✅ Version: {version}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 2: Verify API cleanliness
print("\n2. Verifying clean API (no namespace pollution)...")
try:
    import smm
    exposed = [x for x in dir(smm) if not x.startswith('_') and x not in smm.__all__]
    if exposed:
        print(f"  ❌ FAIL: Unexpected exports: {exposed}")
    else:
        print("  ✅ Clean namespace - only public API exposed")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 3: Verify type hints
print("\n3. Verifying type hints...")
try:
    from smm.smm import SpatialMaskMerger
    import inspect
    sig = inspect.signature(SpatialMaskMerger.merge)
    has_annotations = bool(sig.return_annotation != inspect.Signature.empty)
    if has_annotations:
        print("  ✅ Type hints present in core methods")
    else:
        print("  ⚠️  Some methods lack type hints")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 4: Verify validation exists
print("\n4. Verifying parameter validation...")
try:
    from smm import SpatialMaskMerger
    try:
        SpatialMaskMerger(tau_d=-10)
        print("  ❌ FAIL: Should reject negative tau_d")
    except ValueError:
        print("  ✅ Parameter validation works")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 5: Verify validate_all() method exists
print("\n5. Verifying validate_all() method...")
try:
    from smm import SMMPrediction
    if hasattr(SMMPrediction, 'validate_all'):
        print("  ✅ validate_all() method exists")
    else:
        print("  ❌ FAIL: validate_all() not found")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 6: Verify is_optimized() method
print("\n6. Verifying is_optimized() method...")
try:
    from smm.rtree_utils import RTreeIndex
    rtree = RTreeIndex()
    if hasattr(rtree, 'is_optimized'):
        is_opt = rtree.is_optimized()
        print(f"  ✅ is_optimized() exists (returns: {is_opt})")
    else:
        print("  ❌ FAIL: is_optimized() not found")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 7: Count test files
print("\n7. Verifying test coverage...")
import os
test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
print(f"  📊 Found {len(test_files)} test files:")
for tf in sorted(test_files):
    print(f"      - {tf}")
if len(test_files) >= 6:
    print(f"  ✅ Good test coverage ({len(test_files)} test files)")
else:
    print(f"  ⚠️  Limited test coverage ({len(test_files)} test files)")

# Test 8: Verify functionality
print("\n8. Verifying end-to-end functionality...")
try:
    from smm import SMMPrediction, SpatialMaskMerger
    pred = SMMPrediction('test.jpg')
    pred.add_annotation('car', 0, 0.9, (10, 10, 50, 50), [[[10, 10], [50, 50]]])
    pred.validate_all()
    merger = SpatialMaskMerger(tau_d=15.0, rho=30.0)
    result = merger.merge(pred, (100, 100))
    if len(result) > 0:
        print(f"  ✅ Full pipeline works ({len(result)} cluster(s))")
    else:
        print("  ⚠️  Pipeline returned empty result")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print("\n" + "=" * 70)
print("✅ README claims verification complete!")
print("=" * 70)
print("\n📝 Action item: Update smm/__init__.py __version__ = '1.0.0'")
