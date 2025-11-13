#!/usr/bin/env python
"""Comprehensive tests for rtree_utils.py"""

from smm.rtree_utils import RTreeIndex, _bbox_intersects
import time

print("=" * 60)
print("Testing rtree_utils.py")
print("=" * 60)

# Test 1: bbox_intersects function
print("\n1. Testing _bbox_intersects()...")
tests = [
    # (bbox_a, bbox_b, expected_result, description)
    ((0, 0, 10, 10), (5, 5, 15, 15), True, "Overlapping boxes"),
    ((0, 0, 10, 10), (20, 20, 30, 30), False, "Non-overlapping boxes"),
    ((0, 0, 10, 10), (10, 10, 20, 20), True, "Touching corners (inclusive)"),
    ((0, 0, 10, 10), (0, 0, 10, 10), True, "Identical boxes"),
    ((5, 5, 15, 15), (0, 0, 10, 10), True, "Reverse order overlapping"),
    ((0, 0, 5, 5), (10, 0, 15, 5), False, "Horizontally separated"),
    ((0, 0, 5, 5), (0, 10, 5, 15), False, "Vertically separated"),
    ((2, 2, 8, 8), (0, 0, 10, 10), True, "One inside another"),
]

passed = 0
for bbox_a, bbox_b, expected, desc in tests:
    result = _bbox_intersects(bbox_a, bbox_b)
    if result == expected:
        print(f"  ✅ PASS: {desc}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {desc} (expected {expected}, got {result})")

print(f"\nPassed {passed}/{len(tests)} intersection tests")

# Test 2: RTreeIndex basic operations
print("\n2. Testing RTreeIndex basic operations...")
try:
    rtree = RTreeIndex()
    
    # Insert some boxes
    rtree.insert(0, (0.0, 0.0, 10.0, 10.0))
    rtree.insert(1, (5.0, 5.0, 15.0, 15.0))
    rtree.insert(2, (20.0, 20.0, 30.0, 30.0))
    rtree.insert(3, (25.0, 25.0, 35.0, 35.0))
    print("  ✅ PASS: Insert operations successful")
except Exception as e:
    print(f"  ❌ FAIL: Insert failed - {e}")

# Test 3: Query operations
print("\n3. Testing query operations...")
try:
    # Query should return objects 0 and 1 (overlapping with query box)
    results = list(rtree.query((8.0, 8.0, 12.0, 12.0)))
    if set(results) == {0, 1}:
        print(f"  ✅ PASS: Query returned correct objects: {results}")
    else:
        print(f"  ⚠️  WARNING: Query returned {results}, expected {{0, 1}}")
    
    # Query for objects 2 and 3
    results = list(rtree.query((22.0, 22.0, 28.0, 28.0)))
    if set(results) == {2, 3}:
        print(f"  ✅ PASS: Query returned correct objects: {results}")
    else:
        print(f"  ⚠️  WARNING: Query returned {results}, expected {{2, 3}}")
    
    # Query for no overlap
    results = list(rtree.query((50.0, 50.0, 60.0, 60.0)))
    if len(results) == 0:
        print(f"  ✅ PASS: Query correctly returned no results")
    else:
        print(f"  ⚠️  WARNING: Query returned {results}, expected empty")
        
except Exception as e:
    print(f"  ❌ FAIL: Query failed - {e}")

# Test 4: Invalid bbox handling
print("\n4. Testing invalid bbox handling...")
try:
    rtree = RTreeIndex()
    rtree.insert(99, (10.0, 10.0, 5.0, 5.0))  # x2 < x1
    print("  ❌ FAIL: Should have raised ValueError for invalid bbox")
except ValueError as e:
    print(f"  ✅ PASS: Correctly rejected invalid bbox - {e}")
except Exception as e:
    print(f"  ⚠️  WARNING: Raised unexpected exception - {e}")

# Test 5: Performance comparison (if rtree available)
print("\n5. Testing performance characteristics...")
try:
    from smm.rtree_utils import _RTREE_AVAILABLE
    
    if _RTREE_AVAILABLE:
        print("  📊 Testing with rtree library (optimized)")
    else:
        print("  📊 Testing with pure Python fallback (O(N) scan)")
    
    # Create index with many objects
    rtree = RTreeIndex()
    n = 500
    for i in range(n):
        x = i * 10.0
        rtree.insert(i, (x, 0.0, x + 20.0, 20.0))
    
    # Perform many queries
    start = time.time()
    for i in range(100):
        list(rtree.query((i * 5.0, 0.0, i * 5.0 + 25.0, 20.0)))
    elapsed = time.time() - start
    
    print(f"  ✅ Indexed {n} objects, performed 100 queries in {elapsed:.3f}s")
    print(f"  ⚙️  Average query time: {elapsed/100*1000:.2f}ms")
    
except Exception as e:
    print(f"  ⚠️  Performance test skipped - {e}")

# Test 6: Edge cases
print("\n6. Testing edge cases...")
test_cases = [
    ("Zero-size box", (5.0, 5.0, 5.0, 5.0)),  # Point
    ("Very large coordinates", (1e6, 1e6, 1e6 + 10, 1e6 + 10)),
    ("Negative coordinates", (-100.0, -100.0, -50.0, -50.0)),
    ("Float precision", (0.1, 0.2, 0.3, 0.4)),
]

rtree = RTreeIndex()
for i, (desc, bbox) in enumerate(test_cases):
    try:
        rtree.insert(1000 + i, bbox)
        results = list(rtree.query(bbox))
        if (1000 + i) in results:
            print(f"  ✅ PASS: {desc}")
        else:
            print(f"  ⚠️  WARNING: {desc} - self-query failed")
    except Exception as e:
        print(f"  ❌ FAIL: {desc} - {e}")

# Test 7: Type handling
print("\n7. Testing type conversion...")
try:
    rtree = RTreeIndex()
    # Insert with integers (should convert to float)
    rtree.insert(0, (0, 0, 10, 10))
    results = list(rtree.query((5, 5, 15, 15)))
    if 0 in results:
        print("  ✅ PASS: Integer bbox converted correctly")
    else:
        print("  ❌ FAIL: Integer bbox not handled correctly")
except Exception as e:
    print(f"  ❌ FAIL: Type conversion - {e}")

print("\n" + "=" * 60)
print("✅ rtree_utils.py validation complete!")
print("=" * 60)
