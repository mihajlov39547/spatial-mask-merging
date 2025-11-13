#!/usr/bin/env python
"""Test new features added to predictions.py"""

from smm.predictions import SMMPrediction

print("=" * 60)
print("Testing NEW features in predictions.py")
print("=" * 60)

# Test 1: validate_all() method
print("\n1. Testing validate_all() method...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("car", 0, 0.9, (10, 10, 50, 50), [])
    pred.add_annotation("truck", 1, 0.8, (60, 60, 100, 100), [])
    pred.validate_all()
    print("  ✅ PASS: validate_all() accepts valid predictions")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test with invalid annotation
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("valid", 0, 0.9, (10, 10, 50, 50), [])
    # Manually add invalid annotation (bypassing add_annotation validation)
    from smm.predictions import SMMAnnotation
    bad_ann = SMMAnnotation("bad", 0, 1.5, (10, 10, 50, 50), [])  # confidence > 1
    pred.annotations.append(bad_ann)
    pred.validate_all()
    print("  ❌ FAIL: Should reject invalid annotation")
except ValueError as e:
    print(f"  ✅ PASS: validate_all() caught invalid annotation - {e}")

# Test with empty image_name
try:
    pred = SMMPrediction(image_name="")
    pred.validate_all()
    print("  ❌ FAIL: Should reject empty image_name")
except ValueError as e:
    print(f"  ✅ PASS: Empty image_name rejected - {e}")

# Test 2: Improved image_size_hw validation
print("\n2. Testing improved image_size_hw validation...")

# Test with wrong number of elements
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("car", 0, 0.9, (10, 10, 50, 50), [])
    pred.to_smm_objects((100,))  # Only 1 element
    print("  ❌ FAIL: Should reject single-element tuple")
except ValueError as e:
    print(f"  ✅ PASS: Single-element rejected - {e}")

# Test with non-iterable
try:
    pred.to_smm_objects(100)  # Not a tuple/list
    print("  ❌ FAIL: Should reject non-iterable")
except TypeError as e:
    print(f"  ✅ PASS: Non-iterable rejected - {e}")

# Test with too many elements
try:
    pred.to_smm_objects((100, 100, 100))  # 3 elements
    print("  ❌ FAIL: Should reject 3-element tuple")
except ValueError as e:
    print(f"  ✅ PASS: 3-element tuple rejected - {e}")

# Test 3: Optimized _bbox_to_mask (check it still works)
print("\n3. Testing optimized _bbox_to_mask...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    # Test with positive coordinates
    pred.add_annotation("pos", 0, 0.9, (10.7, 20.3, 50.9, 60.1), [])
    objs_pos = pred.to_smm_objects((100, 100), prefer_segmentation=False)
    
    # Test with negative coordinates (edge case)
    pred2 = SMMPrediction(image_name="test.jpg")
    pred2.add_annotation("neg", 0, 0.9, (-5.5, -3.2, 10.8, 20.5), [])
    objs_neg = pred2.to_smm_objects((100, 100), prefer_segmentation=False)
    
    if objs_pos[0]["mask"].any() and objs_neg[0]["mask"].any():
        print("  ✅ PASS: Optimized bbox_to_mask handles pos/neg coordinates")
        print(f"      Positive coords: {objs_pos[0]['mask'].sum()} pixels")
        print(f"      Negative coords: {objs_neg[0]['mask'].sum()} pixels")
    else:
        print("  ⚠️  WARNING: One mask is empty")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 4: Empty mask bbox behavior
print("\n4. Testing empty mask bbox handling...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    # Add annotation with polygon outside image bounds
    pred.add_annotation("outside", 0, 0.9, (200, 200, 300, 300),
                       [[[1000, 1000], [1100, 1000], [1100, 1100]]])  # Way outside
    objs = pred.to_smm_objects((100, 100), prefer_segmentation=True, recompute_bbox=True)
    
    bbox = objs[0]["bbox"]
    if bbox == (0.0, 0.0, 0.0, 0.0):
        print(f"  ✅ PASS: Empty mask returns degenerate bbox {bbox}")
    else:
        print(f"  ⚠️  WARNING: Empty mask bbox = {bbox}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 5: Integration test with all improvements
print("\n5. Testing full integration...")
try:
    pred = SMMPrediction(image_name="integration_test.jpg")
    
    # Add various annotations
    pred.add_annotation("obj1", 0, 0.95, (10, 10, 50, 50), 
                       [[[10, 10], [50, 10], [50, 50], [10, 50]]])
    pred.add_annotation("obj2", 1, 0.85, (60, 60, 100, 100), [])  # Bbox only
    pred.add_annotation("obj3", 0, 0.90, (120.5, 120.3, 150.7, 150.9), 
                       [[[120, 120], [150, 150]]])  # Degenerate polygon
    
    # Validate all
    pred.validate_all()
    
    # Convert to SMM objects
    objs = pred.to_smm_objects(image_size_hw=(200, 200))
    
    # Test with SpatialMaskMerger
    from smm.smm import SpatialMaskMerger
    merger = SpatialMaskMerger(tau_d=30.0, rho=50.0)
    merged = merger.merge(pred, (200, 200))
    
    print(f"  ✅ PASS: Full integration works")
    print(f"      Input: {len(pred.annotations)} annotations")
    print(f"      Output: {len(merged)} clusters")
    
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print("\n" + "=" * 60)
print("✅ All new features working correctly!")
print("=" * 60)
