#!/usr/bin/env python3
"""
Example execution demonstrating the parallel diversified retrieval strategy.
This simulates the behavior without requiring actual retrieval infrastructure.
"""

from langProBe.hover.hover_program import HoverMultiHop


def simulate_execution():
    """Simulate an execution to show the flow"""

    print("="*70)
    print("HoverMultiHop Parallel Diversified Retrieval - Example Execution")
    print("="*70)

    # Initialize
    program = HoverMultiHop()
    print(f"\n1. Initialized HoverMultiHop with k={program.k}")

    # Example claim
    claim = "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889"
    print(f"\n2. Input Claim:\n   '{claim}'")

    # Simulate the three parallel retrieval strategies
    print("\n3. Three Parallel Retrieval Strategies:")
    print("\n   ┌─────────────────────────────────────────┐")
    print("   │ HOP 1: Direct Claim Retrieval (k=21)    │")
    print("   │ Query: [original claim]                 │")
    print("   └─────────────────────────────────────────┘")
    print("   Expected docs: Direct info about Eiffel Tower, 1889, etc.")

    print("\n   ┌─────────────────────────────────────────┐")
    print("   │ HOP 2: Related Entities (k=21)          │")
    print("   │ Query: 'related entities, people, or    │")
    print("   │         works mentioned in: {claim}'    │")
    print("   └─────────────────────────────────────────┘")
    print("   Expected docs: Gustave Eiffel bio, Paris landmarks, etc.")

    print("\n   ┌─────────────────────────────────────────┐")
    print("   │ HOP 3: Background Context (k=21)        │")
    print("   │ Query: 'background information and      │")
    print("   │         context about: {claim}'         │")
    print("   └─────────────────────────────────────────┘")
    print("   Expected docs: 19th century Paris, World Expo 1889, etc.")

    print("\n4. Total Retrieved: 63 documents (21 from each hop)")

    # Simulate diverse document set
    simulated_docs = [
        # Direct retrieval results
        "Eiffel Tower | The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Eiffel Tower | Completed in 1889, designed by Gustave Eiffel.",
        "Eiffel Tower | Built for the 1889 World's Fair in Paris.",
        "Eiffel Tower | Stands 330 meters tall on the Champ de Mars.",
        "Eiffel Tower | Most visited paid monument in the world.",

        # Entity-focused results
        "Gustave Eiffel | French civil engineer born in 1832.",
        "Gustave Eiffel | Also designed the Statue of Liberty framework.",
        "Gustave Eiffel | Founded engineering consultancy in Paris.",
        "Maurice Koechlin | Engineer who made initial Eiffel Tower drawings.",
        "Stephen Sauvestre | Architect who refined the tower's design.",

        # Context results
        "1889 World's Fair | Universal Exposition held in Paris.",
        "1889 World's Fair | Celebrated 100th anniversary of French Revolution.",
        "Paris landmarks | Major historical monuments of the French capital.",
        "Iron architecture | 19th century engineering innovations.",
        "French Revolution | Centennial celebrated with the 1889 exposition.",

        # Duplicates and near-duplicates (will be filtered)
        "Eiffel Tower | The Eiffel Tower was built in 1889.",  # Near duplicate
        "EIFFEL TOWER | The tower is located in Paris, France.",  # Case variant
        "Eiffel Tower | Completed in 1889, designed by Gustave Eiffel.",  # Exact dup

        # Additional diverse documents
        "Champ de Mars | Public park in Paris, location of Eiffel Tower.",
        "Wrought iron | Material used in 19th century construction.",
        "Lattice structures | Engineering designs for tall structures.",
    ]

    print("\n5. Diversity-Based Reranking:")
    print("   - Removing exact duplicates...")
    print("   - Computing TF-IDF vectors...")
    print("   - Calculating relevance to claim...")
    print("   - Applying MMR algorithm (λ=0.5)...")

    result = program._diversified_rerank(simulated_docs, claim, top_k=21)

    print(f"\n6. Final Output: {len(result)} diverse documents")
    print("\n   Selected documents (ranked by MMR):")

    for i, doc in enumerate(result, 1):
        # Truncate for display
        doc_display = doc if len(doc) <= 60 else doc[:57] + "..."
        print(f"   {i:2d}. {doc_display}")

    print("\n7. Output Characteristics:")
    print(f"   ✓ Total documents: {len(result)}")
    print(f"   ✓ All unique: {len(result) == len(set(result))}")
    print(f"   ✓ Meets constraint: {len(result) <= 21}")

    print("\n" + "="*70)
    print("Key Benefits Demonstrated:")
    print("="*70)
    print("✓ No information loss (no summarization)")
    print("✓ Parallel retrieval strategies (3 complementary views)")
    print("✓ Smart diversity selection (MMR avoids redundancy)")
    print("✓ Meets all constraints (≤21 docs, 3 searches)")
    print("="*70)


if __name__ == "__main__":
    simulate_execution()
