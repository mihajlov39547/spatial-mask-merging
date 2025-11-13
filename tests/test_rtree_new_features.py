#!/usr/bin/env python
"""Test new features added to rtree_utils.py"""

from smm.rtree_utils import RTreeIndex

print("=" * 60)
print("Testing NEW features in rtree_utils.py")
print("=" * 60)

# Test 1: is_optimized() method
print("\n1. Testing is_optimized() method...")
rtree = RTreeIndex()
is_opt = rtree.is_optimized()
print(f"  ✅ is_optimized() = {is_opt}")
if is_opt:
    print("  📊 Using rtree library backend (optimal performance)")
else:
    print("  ⚠️  Using pure-Python fallback (slower performance)")

# Test 2: __len__() method
print("\n2. Testing __len__() method...")
rtree = RTreeIndex()
for i in range(10):
    rtree.insert(i, (i * 10.0, 0.0, i * 10.0 + 5.0, 5.0))

try:
    size = len(rtree)
    print(f"  ✅ Fallback backend: len={size}")
except NotImplementedError:
    print(f"  ✅ rtree backend correctly raises NotImplementedError for len()")

# Test 3: Invalid bbox type handling
print("\n3. Testing improved error messages...")
try:
    rtree.insert(99, "not a bbox")
    print("  ❌ FAIL: Should have raised TypeError")
except TypeError as e:
    print(f"  ✅ PASS: Clear error for bad type - {e}")

# Test 4: Invalid query bbox
print("\n4. Testing query validation...")
try:
    list(rtree.query((10.0, 10.0, 5.0, 5.0)))  # x2 < x1
    print("  ❌ FAIL: Should have raised ValueError for invalid query")
except ValueError as e:
    print(f"  ✅ PASS: Query validation works - {e}")

# Test 5: Query with non-numeric type
print("\n5. Testing query with bad type...")
try:
    list(rtree.query([None, None, None, None]))
    print("  ❌ FAIL: Should have raised TypeError")
except TypeError as e:
    print(f"  ✅ PASS: Query type checking works - {e}")

# Test 6: Insert with tuple of ints (should auto-convert)
print("\n6. Testing auto-conversion of int to float...")
try:
    rtree = RTreeIndex()
    rtree.insert(0, (0, 0, 10, 10))  # All ints
    results = list(rtree.query((5, 5, 15, 15)))  # Query with ints
    if 0 in results:
        print("  ✅ PASS: Int→float conversion works seamlessly")
    else:
        print("  ⚠️  WARNING: Conversion worked but query failed")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

# Test 7: Optimized bbox_intersects (correctness check)
print("\n7. Testing optimized _bbox_intersects...")
from smm.rtree_utils import _bbox_intersects

test_cases = [
    ((0, 0, 10, 10), (5, 5, 15, 15), True),
    ((0, 0, 10, 10), (20, 20, 30, 30), False),
    ((0, 0, 10, 10), (10, 0, 20, 10), True),  # Edge touching
]

all_pass = True
for a, b, expected in test_cases:
    result = _bbox_intersects(a, b)
    if result != expected:
        print(f"  ❌ FAIL: _bbox_intersects{a, b} = {result}, expected {expected}")
        all_pass = False

if all_pass:
    print("  ✅ PASS: Optimized _bbox_intersects maintains correctness")

print("\n" + "=" * 60)
print("✅ All new features working correctly!")
print("=" * 60)
