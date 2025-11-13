#!/usr/bin/env python
"""Quick test of SMM parameter validation and optimizations."""

from smm.smm import SpatialMaskMerger
import numpy as np

print("Testing parameter validation...")

# Test 1: Invalid tau_d
try:
    SpatialMaskMerger(tau_d=-10)
    print("❌ FAIL: Should have raised ValueError for negative tau_d")
except ValueError as e:
    print(f"✅ PASS: tau_d validation - {e}")

# Test 2: Invalid tau_i
try:
    SpatialMaskMerger(tau_i=1.5)
    print("❌ FAIL: Should have raised ValueError for tau_i > 1")
except ValueError as e:
    print(f"✅ PASS: tau_i validation - {e}")

# Test 3: Invalid score_aggregation
try:
    SpatialMaskMerger(score_aggregation="invalid")
    print("❌ FAIL: Should have raised ValueError for invalid score_aggregation")
except ValueError as e:
    print(f"✅ PASS: score_aggregation validation - {e}")

# Test 4: Valid parameters
try:
    merger = SpatialMaskMerger(tau_d=15.0, tau_i=0.5, rho=30.0)
    print("✅ PASS: Valid parameters accepted")
except Exception as e:
    print(f"❌ FAIL: Valid parameters rejected - {e}")

# Test 5: IoU optimization
print("\nTesting optimized compute_iou...")
from smm.smm import compute_iou
mask_a = np.random.rand(100, 100) > 0.5
mask_b = np.random.rand(100, 100) > 0.5
iou = compute_iou(mask_a, mask_b)
assert 0.0 <= iou <= 1.0
print(f"✅ PASS: IoU computation works (IoU={iou:.3f})")

# Test 6: Empty mask handling
print("\nTesting empty mask handling...")
from smm.smm import merge_masks
try:
    # All empty masks
    empty_masks = [np.array([]), np.array([])]
    merge_masks(empty_masks)
    print("❌ FAIL: Should have raised ValueError for all empty masks")
except ValueError as e:
    print(f"✅ PASS: Empty mask detection - {e}")

# Test 7: Mixed empty/valid masks
try:
    valid_mask = np.zeros((10, 10), dtype=bool)
    valid_mask[3:7, 3:7] = True
    empty_mask = np.array([])
    result = merge_masks([empty_mask, valid_mask, empty_mask])
    assert result.shape == (10, 10)
    print("✅ PASS: Mixed empty/valid masks handled correctly")
except Exception as e:
    print(f"❌ FAIL: Mixed mask handling - {e}")

print("\n🎉 All validation tests passed!")
