#!/usr/bin/env python3
"""
Example demonstrating how different types of claims would be analyzed
and allocated k-values in the HoverMultiHopPredict system.
"""

from langProBe.hover.hover_program import HoverMultiHopPredict

def analyze_claim_example(claim_text, expected_entities, expected_complexity):
    """Show how a claim would be analyzed and allocated k-values."""
    program = HoverMultiHopPredict()

    print(f"Claim: \"{claim_text}\"")
    print(f"  Expected entities: {expected_entities}")

    # Get k-value allocation based on expected entity count
    k1, k2, k3 = program._allocate_k_values(expected_entities)

    print(f"  Detected complexity: {expected_complexity}")
    print(f"  K-value allocation: Hop1={k1}, Hop2={k2}, Hop3={k3}")
    print(f"  Total documents: {k1 + k2 + k3}")

    # Explain the retrieval strategy
    if expected_entities <= 2:
        print(f"  📚 Strategy: Deep focus on primary entity")
        print(f"     - Hop 1: Retrieve {k1} docs about the main topic")
        print(f"     - Hop 2: Retrieve {k2} docs to verify specific details")
        print(f"     - Hop 3: Retrieve {k3} docs for final confirmation")
    elif expected_entities == 3:
        print(f"  ⚖️  Strategy: Balanced multi-hop exploration")
        print(f"     - Hop 1: Retrieve {k1} docs for initial entities")
        print(f"     - Hop 2: Retrieve {k2} docs for relationships")
        print(f"     - Hop 3: Retrieve {k3} docs for cross-verification")
    else:
        print(f"  🌐 Strategy: Broad multi-entity coverage")
        print(f"     - Hop 1: Retrieve {k1} docs to cast wide net")
        print(f"     - Hop 2: Retrieve {k2} docs to explore connections")
        print(f"     - Hop 3: Retrieve {k3} docs to verify relationships")

    print()

# Example claims of varying complexity
print("=" * 80)
print("Adaptive K-Value Allocation: Example Claims")
print("=" * 80)
print()

print("SIMPLE CLAIMS (1-2 entities)")
print("-" * 80)
analyze_claim_example(
    claim_text="Paris is the capital of France.",
    expected_entities=2,  # Paris, France
    expected_complexity="Simple"
)

analyze_claim_example(
    claim_text="Albert Einstein developed the theory of relativity.",
    expected_entities=2,  # Einstein, relativity theory
    expected_complexity="Simple"
)

print("\nMODERATE CLAIMS (3 entities)")
print("-" * 80)
analyze_claim_example(
    claim_text="The Eiffel Tower was built for the 1889 World's Fair in Paris.",
    expected_entities=3,  # Eiffel Tower, World's Fair 1889, Paris
    expected_complexity="Moderate"
)

analyze_claim_example(
    claim_text="Barack Obama served as the 44th President of the United States.",
    expected_entities=3,  # Obama, 44th President, United States
    expected_complexity="Moderate"
)

print("\nCOMPLEX CLAIMS (4+ entities)")
print("-" * 80)
analyze_claim_example(
    claim_text="Marie Curie, born in Warsaw, won Nobel Prizes in both Physics and Chemistry while working in Paris.",
    expected_entities=5,  # Curie, Warsaw, Nobel Physics, Nobel Chemistry, Paris
    expected_complexity="Complex"
)

analyze_claim_example(
    claim_text="The Apollo 11 mission, launched by NASA in 1969, successfully landed astronauts Neil Armstrong and Buzz Aldrin on the Moon.",
    expected_entities=5,  # Apollo 11, NASA, 1969, Armstrong, Aldrin
    expected_complexity="Complex"
)

analyze_claim_example(
    claim_text="During World War II, Winston Churchill served as British Prime Minister while Franklin D. Roosevelt was US President and Joseph Stalin led the Soviet Union.",
    expected_entities=5,  # WWII, Churchill, UK, FDR, USA, Stalin, USSR (7 entities, capped at 5)
    expected_complexity="Complex"
)

print("=" * 80)
print("Summary")
print("=" * 80)
print("The adaptive k-value allocation ensures that:")
print("  • Simple claims get deep coverage of the main entity early")
print("  • Moderate claims get balanced exploration across related entities")
print("  • Complex claims get broad initial coverage with deeper later hops")
print("  • Total budget remains constant at 21 documents across all strategies")
print("=" * 80)
