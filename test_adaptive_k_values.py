#!/usr/bin/env python3
"""
Test script demonstrating the adaptive k-value allocation strategy in HoverMultiHopPredict.

This script shows how different claim complexities result in different k-value distributions
across the 3 retrieval hops.
"""

from langProBe.hover.hover_program import HoverMultiHopPredict

# Test the k-value allocation strategy
program = HoverMultiHopPredict()

print("=" * 80)
print("Adaptive K-Value Allocation Strategy Test")
print("=" * 80)
print("\nTotal budget: 21 documents across 3 hops")
print()

# Test different complexity levels
test_cases = [
    (1, "Simple claim (1 entity)"),
    (2, "Simple claim (2 entities)"),
    (3, "Moderate claim (3 entities)"),
    (4, "Complex claim (4 entities)"),
    (5, "Complex claim (5+ entities)"),
]

for num_entities, description in test_cases:
    k1, k2, k3 = program._allocate_k_values(num_entities)
    total = k1 + k2 + k3

    print(f"{description}:")
    print(f"  num_entities = {num_entities}")
    print(f"  k-values = [{k1}, {k2}, {k3}]")
    print(f"  total = {total} ✓" if total == 21 else f"  total = {total} ✗ ERROR")

    # Explain the strategy
    if num_entities <= 2:
        strategy = "Go deep early - focus on main entity"
    elif num_entities == 3:
        strategy = "Balanced coverage across all hops"
    else:
        strategy = "Cast wider net - explore multiple entities across hops"
    print(f"  Strategy: {strategy}")
    print()

print("=" * 80)
print("Strategy Summary:")
print("=" * 80)
print("Simple claims (1-2 entities):   k=[10, 8, 3] - Deep dive on primary entity")
print("Moderate claims (3 entities):   k=[7, 7, 7] - Balanced multi-hop exploration")
print("Complex claims (4+ entities):   k=[5, 8, 8] - Broad initial search, deeper later")
print()
print("All allocations sum to exactly 21 documents to maintain budget consistency.")
print("=" * 80)
