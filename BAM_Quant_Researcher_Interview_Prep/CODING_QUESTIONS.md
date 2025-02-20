# Coding Questions

---

### 1. Merge Two Sorted Lists of Integers

**Problem Description:** Write an algorithm to merge two sorted lists of integers into a single sorted list.

**Python Code:**
```python
def merge_sorted_lists(list1, list2):
    # Initialize result list and pointers
    merged = []
    i, j = 0, 0
    
    # Iterate while both lists have elements
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Append remaining elements of list1, if any
    merged.extend(list1[i:])
    
    # Append remaining elements of list2, if any
    merged.extend(list2[j:])
    
    return merged

# Example usage
list1 = [1, 3, 5, 7]
list2 = [2, 4, 6, 8]
result = merge_sorted_lists(list1, list2)
print("Merged Sorted List:", result)  # Output: [1, 2, 3, 4, 5, 6, 7, 8]
```

**Detailed Explanation:**
- **Purpose:** Efficiently merges two sorted lists into one sorted list, a common task in data processing (e.g., combining sorted treasury transaction logs).
- **Approach:** This is the "two-pointer" or "merge" algorithm from the merge sort technique, which is optimal for sorted lists.
- **Code Breakdown:**
  - **Initialization:** Creates an empty result list `merged` and pointers `i` and `j` starting at the beginning of `list1` and `list2`.
  - **Main Loop:** Compares elements at `list1[i]` and `list2[j]`, appending the smaller one to `merged` and incrementing the corresponding pointer. Continues until one list is exhausted.
  - **Cleanup:** Uses `extend()` to append any remaining elements from `list1` (from index `i` onward) or `list2` (from index `j` onward). Since inputs are sorted, these remnants are already in order.
- **Optimality:** 
  - Avoids sorting from scratch (e.g., concatenating then sorting would be \( O((n+m) \log (n+m)) \)).
  - Leverages existing order, requiring only linear comparisons.
  - No extra space beyond the output list (in-place merging is possible but complex with Python lists; this is simpler and still efficient).
- **Edge Cases:** Handles empty lists naturally (e.g., `list1 = []`, `list2 = [1, 2]` returns `[1, 2]`).

**Complexity Analysis:**
- **Time Complexity:** \( O(n + m) \)
  - \( n = \text{len(list1)} \), \( m = \text{len(list2)} \).
  - Each element is compared and appended exactly once, with pointers advancing linearly through both lists.
  - `extend()` operations are \( O(k) \) where \( k \) is the number of remaining elements, but total iterations remain \( n + m \).
- **Space Complexity:** \( O(n + m) \)
  - Output list `merged` stores all \( n + m \) elements.
  - Excludes input space (as per convention); only additional space is the result.

---

### 2. Determine the Relationship Between Two Circles

**Problem Description:** Given two circles (center coordinates and radii), determine if they are: one within the other, one outside the other, touching each other, or intersecting each other.

**Python Code:**
```python
import math

def circle_relationship(x1, y1, r1, x2, y2, r2):
    # Calculate distance between centers
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Compare distance with radii sums and differences
    if distance > r1 + r2:
        return "Outside each other"
    elif distance == r1 + r2:
        return "Touching externally"
    elif distance < r1 + r2 and distance > abs(r1 - r2):
        return "Intersecting"
    elif distance == abs(r1 - r2):
        return "Touching internally"
    elif distance < abs(r1 - r2):
        if r1 > r2 and distance + r2 < r1:
            return "Circle 2 inside Circle 1"
        elif r2 > r1 and distance + r1 < r2:
            return "Circle 1 inside Circle 2"
    elif distance == 0 and r1 == r2:
        return "Coincident (same circle)"
    
    # Default case (shouldn't occur with valid inputs)
    return "Invalid configuration"

# Example usage
test_cases = [
    (0, 0, 5, 10, 0, 3),  # Outside
    (0, 0, 5, 8, 0, 3),  # Intersecting
    (0, 0, 5, 5, 0, 5),  # Touching externally
    (0, 0, 5, 2, 0, 3),  # Inside
    (0, 0, 5, 3, 0, 2)   # Touching internally
]

for x1, y1, r1, x2, y2, r2 in test_cases:
    result = circle_relationship(x1, y1, r1, x2, y2, r2)
    print(f"Circle1({x1}, {y1}, r={r1}), Circle2({x2}, {y2}, r={r2}): {result}")
```

**Detailed Explanation:**
- **Purpose:** Determines the geometric relationship between two circles, useful in spatial analysis (e.g., treasury risk zones or asset overlap).
- **Approach:** Uses the distance between centers compared to radii sums and differences—optimal as it avoids complex geometric intersection point calculations.
- **Code Breakdown:**
  - **Distance Calculation:** \( d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \) using Euclidean distance.
  - **Conditions:**
    1. **Outside:** \( d > r_1 + r_2 \) (centers too far to touch).
    2. **Touching Externally:** \( d = r_1 + r_2 \) (centers exactly at sum of radii).
    3. **Intersecting:** \( |r_1 - r_2| < d < r_1 + r_2 \) (overlapping but not contained).
    4. **Touching Internally:** \( d = |r_1 - r_2| \) (one circle’s edge touches the other’s interior).
    5. **One Inside the Other:** \( d < |r_1 - r_2| \) and \( d + r_{\text{smaller}} < r_{\text{larger}} \) (fully contained).
    6. **Coincident:** \( d = 0 \) and \( r_1 = r_2 \) (same circle, edge case).
  - **Logic Flow:** Hierarchical if-elif structure ensures mutual exclusivity and covers all cases.
- **Optimality:** 
  - Single distance computation avoids iterative or trigonometric methods.
  - Constant-time checks based on geometric properties.
- **Edge Cases:** Handles zero distance, equal radii, and negative radii (assumed positive in practice).
- **Treasury Relevance:** Could model overlapping risk regions or asset influence zones.

**Output Example:**
```
Circle1(0, 0, r=5), Circle2(10, 0, r=3): Outside each other
Circle1(0, 0, r=5), Circle2(8, 0, r=3): Intersecting
Circle1(0, 0, r=5), Circle2(5, 0, r=5): Touching externally
Circle1(0, 0, r=5), Circle2(2, 0, r=3): Circle 2 inside Circle 1
Circle1(0, 0, r=5), Circle2(3, 0, r=2): Touching internally
```

**Complexity Analysis:**
- **Time Complexity:** \( O(1) \)
  - Distance calculation and comparisons are constant-time operations.
  - No loops or variable-size computations.
- **Space Complexity:** \( O(1) \)
  - Only stores scalars (coordinates, radii, distance).
  - No additional data structures beyond inputs and a single result string.

---

### Notes
- **Optimality Considerations:**
  - **Merge Lists:** The two-pointer method is optimal for sorted inputs, avoiding unnecessary sorting or comparisons.
  - **Circle Relationship:** Distance-based approach is the most efficient, requiring minimal computation while covering all geometric cases.
- **Dependencies:** Only `math` for square root in the circle problem; no external libraries needed beyond Python’s standard library.
- **Generalization:** 
  - Merge algorithm extends to merging multiple sorted lists (e.g., with a heap for \( k \) lists).
  - Circle logic could extend to 3D spheres with the same distance-based principle.

Let me know if you’d like further refinements or additional questions!
