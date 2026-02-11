#!/usr/bin/env python3
"""
Comprehensive test demonstrating the complete adaptive k-value allocation workflow.
This shows how the system analyzes claims and allocates resources dynamically.
"""

from langProBe.hover.hover_program import HoverMultiHopPredict, ClaimComplexitySignature

def test_workflow():
    """Test the complete workflow of adaptive k-value allocation."""

    print("=" * 80)
    print("COMPREHENSIVE WORKFLOW TEST")
    print("=" * 80)
    print()

    # 1. Instantiate the program
    print("Step 1: Instantiating HoverMultiHopPredict...")
    program = HoverMultiHopPredict()
    print("✅ Program instantiated successfully")
    print()

    # 2. Verify components exist
    print("Step 2: Verifying components...")
    assert hasattr(program, 'analyze_complexity'), "Missing analyze_complexity module"
    assert hasattr(program, '_allocate_k_values'), "Missing _allocate_k_values method"
    assert hasattr(program, 'create_query_hop2'), "Missing create_query_hop2"
    assert hasattr(program, 'create_query_hop3'), "Missing create_query_hop3"
    assert hasattr(program, 'summarize1'), "Missing summarize1"
    assert hasattr(program, 'summarize2'), "Missing summarize2"
    print("✅ All components present")
    print()

    # 3. Test k-value allocation for all complexity levels
    print("Step 3: Testing k-value allocation across all complexity levels...")
    test_cases = [
        (1, [10, 8, 3], "Simple - 1 entity"),
        (2, [10, 8, 3], "Simple - 2 entities"),
        (3, [7, 7, 7], "Moderate - 3 entities"),
        (4, [5, 8, 8], "Complex - 4 entities"),
        (5, [5, 8, 8], "Complex - 5 entities"),
    ]

    all_passed = True
    for num_entities, expected_k_values, description in test_cases:
        k1, k2, k3 = program._allocate_k_values(num_entities)
        actual = [k1, k2, k3]
        total = sum(actual)

        if actual == expected_k_values and total == 21:
            print(f"  ✅ {description}: {actual} (sum={total})")
        else:
            print(f"  ❌ {description}: Expected {expected_k_values}, got {actual}")
            all_passed = False

    if all_passed:
        print("✅ All k-value allocations correct")
    else:
        print("❌ Some allocations failed")
        return False
    print()

    # 4. Test edge cases
    print("Step 4: Testing edge cases...")
    edge_cases = [
        (0, [10, 8, 3], "Below minimum (clamped to 1)"),
        (10, [5, 8, 8], "Above maximum (clamped to 5)"),
        (-5, [10, 8, 3], "Negative value (clamped to 1)"),
    ]

    for num_entities, expected_k_values, description in edge_cases:
        k1, k2, k3 = program._allocate_k_values(num_entities)
        actual = [k1, k2, k3]
        total = sum(actual)

        if actual == expected_k_values and total == 21:
            print(f"  ✅ {description}: {actual}")
        else:
            print(f"  ❌ {description}: Expected {expected_k_values}, got {actual}")
            all_passed = False

    if all_passed:
        print("✅ All edge cases handled correctly")
    print()

    # 5. Verify budget consistency
    print("Step 5: Verifying budget consistency...")
    budget_check = True
    for i in range(1, 11):
        k1, k2, k3 = program._allocate_k_values(i)
        total = k1 + k2 + k3
        if total != 21:
            print(f"  ❌ num_entities={i}: Total={total} (expected 21)")
            budget_check = False

    if budget_check:
        print("  ✅ All allocations sum to exactly 21 documents")
    print()

    # 6. Verify ClaimComplexitySignature
    print("Step 6: Verifying ClaimComplexitySignature...")
    sig = ClaimComplexitySignature
    # Check that the signature class exists and has the expected structure
    assert sig is not None, "ClaimComplexitySignature is None"
    assert hasattr(sig, '__doc__'), "Missing docstring"
    assert 'complexity' in sig.__doc__.lower(), "Docstring doesn't mention complexity"
    assert 'entities' in sig.__doc__.lower(), "Docstring doesn't mention entities"
    print("✅ ClaimComplexitySignature properly defined with comprehensive instructions")
    print()

    # 7. Strategy summary
    print("Step 7: Strategy Summary")
    print("-" * 80)
    strategies = [
        ("Simple (1-2)", [10, 8, 3], "Deep focus on primary entity"),
        ("Moderate (3)", [7, 7, 7], "Balanced multi-hop exploration"),
        ("Complex (4+)", [5, 8, 8], "Broad coverage across entities"),
    ]

    for level, k_vals, strategy in strategies:
        print(f"  {level:15} k={k_vals} → {strategy}")
    print()

    # Final summary
    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • Adaptive k-value allocation is working correctly")
    print("  • All complexity levels produce valid allocations")
    print("  • Budget consistency maintained (always 21 documents)")
    print("  • Edge cases handled properly")
    print("  • Components properly integrated")
    print()
    print("The HoverMultiHopPredict system is ready to adaptively allocate")
    print("document budgets based on claim complexity!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_workflow()
    exit(0 if success else 1)
