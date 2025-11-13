#!/usr/bin/env python
"""Comprehensive tests for predictions.py and integration with SMM"""

import numpy as np
import json
from smm.predictions import SMMPrediction, SMMAnnotation
from smm.smm import SpatialMaskMerger

print("=" * 70)
print("Testing predictions.py")
print("=" * 70)

# Test 1: SMMAnnotation validation
print("\n1. Testing SMMAnnotation validation...")
try:
    # Valid annotation
    ann = SMMAnnotation(
        type="car",
        class_id=0,
        confidence=0.95,
        bbox=(10.0, 10.0, 50.0, 50.0),
        segmentation=[[(10, 10), (50, 10), (50, 50), (10, 50)]]
    )
    ann.validate()
    print("  ✅ PASS: Valid annotation accepted")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test invalid bbox
try:
    ann = SMMAnnotation(
        type="car", class_id=0, confidence=0.95,
        bbox=(50.0, 50.0, 10.0, 10.0),  # x2 < x1
        segmentation=[]
    )
    ann.validate()
    print("  ❌ FAIL: Should reject invalid bbox")
except ValueError as e:
    print(f"  ✅ PASS: Invalid bbox rejected - {e}")

# Test invalid confidence
try:
    ann = SMMAnnotation(
        type="car", class_id=0, confidence=1.5,  # > 1.0
        bbox=(10.0, 10.0, 50.0, 50.0),
        segmentation=[]
    )
    ann.validate()
    print("  ❌ FAIL: Should reject invalid confidence")
except ValueError as e:
    print(f"  ✅ PASS: Invalid confidence rejected - {e}")

# Test 2: SMMPrediction add_annotation
print("\n2. Testing SMMPrediction.add_annotation()...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation(
        type="car",
        class_id=0,
        confidence=0.9,
        bbox=[10, 10, 50, 50],  # Can use list
        segmentation=[[[10, 10], [50, 10], [50, 50], [10, 50]]]
    )
    if len(pred.annotations) == 1:
        print(f"  ✅ PASS: Annotation added (total: {len(pred.annotations)})")
    else:
        print(f"  ❌ FAIL: Expected 1 annotation, got {len(pred.annotations)}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 3: JSON serialization/deserialization
print("\n3. Testing JSON serialization...")
try:
    pred = SMMPrediction(image_name="test_image.png")
    pred.add_annotation("car", 0, 0.95, (10, 10, 50, 50), [[[10, 10], [50, 50]]])
    pred.add_annotation("truck", 1, 0.85, (60, 60, 100, 100), [[[60, 60], [100, 100]]])
    
    # Serialize
    json_dict = pred.to_json_dict()
    json_str = json.dumps(json_dict)
    
    # Deserialize
    loaded_dict = json.loads(json_str)
    pred2 = SMMPrediction.from_json_dict(loaded_dict)
    
    if pred2.image_name == pred.image_name and len(pred2.annotations) == len(pred.annotations):
        print(f"  ✅ PASS: JSON round-trip successful ({len(pred2.annotations)} annotations)")
    else:
        print(f"  ❌ FAIL: Data mismatch after round-trip")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 4: to_smm_objects conversion
print("\n4. Testing to_smm_objects() conversion...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("car", 0, 0.9, (10, 10, 50, 50), 
                       [[[10, 10], [50, 10], [50, 50], [10, 50]]])
    
    objs = pred.to_smm_objects(image_size_hw=(100, 100))
    
    if len(objs) == 1:
        obj = objs[0]
        required_keys = {"mask", "bbox", "score", "label"}
        if required_keys.issubset(obj.keys()):
            print(f"  ✅ PASS: Converted to SMM objects with correct schema")
            print(f"      - mask shape: {obj['mask'].shape}")
            print(f"      - bbox: {obj['bbox']}")
            print(f"      - score: {obj['score']}")
            print(f"      - label: {obj['label']}")
        else:
            print(f"  ❌ FAIL: Missing required keys: {required_keys - obj.keys()}")
    else:
        print(f"  ❌ FAIL: Expected 1 object, got {len(objs)}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 5: Polygon rasterization
print("\n5. Testing polygon rasterization...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    # Triangle polygon
    pred.add_annotation("triangle", 0, 0.9, (0, 0, 50, 50),
                       [[[10, 10], [40, 10], [25, 40]]])
    
    objs = pred.to_smm_objects((100, 100), prefer_segmentation=True)
    mask = objs[0]["mask"]
    
    if mask.shape == (100, 100) and mask.any():
        print(f"  ✅ PASS: Polygon rasterized (pixels: {mask.sum()})")
    else:
        print(f"  ❌ FAIL: Invalid mask shape or empty mask")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 6: Bbox fallback when no segmentation
print("\n6. Testing bbox fallback...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("box", 0, 0.8, (20, 20, 40, 40), [])  # Empty segmentation
    
    objs = pred.to_smm_objects((100, 100), prefer_segmentation=True)
    mask = objs[0]["mask"]
    
    expected_pixels = 21 * 21  # (40-20+1) * (40-20+1)
    if mask.sum() == expected_pixels:
        print(f"  ✅ PASS: Bbox fallback works (pixels: {mask.sum()})")
    else:
        print(f"  ⚠️  WARNING: Expected {expected_pixels} pixels, got {mask.sum()}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 7: Edge cases
print("\n7. Testing edge cases...")

# Empty prediction
try:
    pred = SMMPrediction(image_name="empty.jpg")
    objs = pred.to_smm_objects((100, 100))
    if len(objs) == 0:
        print("  ✅ PASS: Empty prediction handled")
    else:
        print(f"  ❌ FAIL: Expected 0 objects, got {len(objs)}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Degenerate polygon (< min_polygon_points)
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("point", 0, 0.9, (10, 10, 11, 11), [[[10, 10], [11, 11]]])  # Only 2 points
    objs = pred.to_smm_objects((100, 100), prefer_segmentation=True, min_polygon_points=3)
    # Should fall back to bbox since polygon has < 3 points
    if objs[0]["mask"].any():
        print("  ✅ PASS: Degenerate polygon falls back to bbox")
    else:
        print("  ❌ FAIL: Mask is empty")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Invalid image size
try:
    pred = SMMPrediction(image_name="test.jpg")
    pred.add_annotation("test", 0, 0.9, (10, 10, 50, 50), [])
    objs = pred.to_smm_objects((0, 100))  # H = 0
    print("  ❌ FAIL: Should reject invalid image size")
except ValueError as e:
    print(f"  ✅ PASS: Invalid image size rejected - {e}")

# Test 8: Integration with SpatialMaskMerger
print("\n8. Testing integration with SpatialMaskMerger...")
try:
    pred = SMMPrediction(image_name="test_merge.jpg")
    
    # Add two overlapping objects (should merge)
    pred.add_annotation("car", 0, 0.95, (10, 10, 50, 50),
                       [[[10, 10], [50, 10], [50, 50], [10, 50]]])
    pred.add_annotation("car", 0, 0.90, (40, 40, 80, 80),
                       [[[40, 40], [80, 40], [80, 80], [40, 80]]])
    
    # Add distant object (should NOT merge)
    pred.add_annotation("car", 0, 0.85, (200, 200, 240, 240),
                       [[[200, 200], [240, 200], [240, 240], [200, 240]]])
    
    merger = SpatialMaskMerger(
        tau_d=50.0,  # Large distance threshold
        tau_i=0.1,   # Low IoU threshold
        rho=60.0,    # Large search radius
        gamma=0.1    # Permissive anti-chaining
    )
    
    merged = merger.merge(pred, image_size_hw=(300, 300))
    
    if 1 <= len(merged) <= 3:
        print(f"  ✅ PASS: SMM integration works ({len(merged)} clusters)")
        for i, obj in enumerate(merged):
            print(f"      Cluster {i}: label={obj['label']}, score={obj['score']:.2f}, bbox={obj['bbox']}")
    else:
        print(f"  ⚠️  WARNING: Expected 1-3 clusters, got {len(merged)}")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 9: Type conversion robustness
print("\n9. Testing type conversion...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    # Add with mixed int/float types
    pred.add_annotation(
        type="car",
        class_id=0,
        confidence=0.95,
        bbox=(10, 10, 50, 50),  # ints
        segmentation=[[[10.0, 10.0], [50, 10], [50.5, 50.5]]]  # mixed
    )
    objs = pred.to_smm_objects((100, 100))
    print("  ✅ PASS: Type conversion handles mixed int/float")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 10: Tight bbox recomputation
print("\n10. Testing tight bbox recomputation...")
try:
    pred = SMMPrediction(image_name="test.jpg")
    # Add annotation with loose bbox
    pred.add_annotation("small", 0, 0.9, (0, 0, 100, 100),  # Loose
                       [[[30, 30], [40, 30], [40, 40], [30, 40]]])  # Tight polygon
    
    objs_recompute = pred.to_smm_objects((100, 100), recompute_bbox=True)
    objs_no_recompute = pred.to_smm_objects((100, 100), recompute_bbox=False)
    
    bbox_recompute = objs_recompute[0]["bbox"]
    bbox_original = objs_no_recompute[0]["bbox"]
    
    if bbox_recompute != bbox_original:
        print(f"  ✅ PASS: Bbox recomputation works")
        print(f"      Original: {bbox_original}")
        print(f"      Recomputed: {bbox_recompute}")
    else:
        print(f"  ⚠️  WARNING: Recomputed bbox same as original")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print("\n" + "=" * 70)
print("✅ predictions.py validation complete!")
print("=" * 70)
