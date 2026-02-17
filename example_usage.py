#!/usr/bin/env python3
"""
Example usage of the refactored HoverMultiHop with entity-extraction-first approach.

This demonstrates how the new implementation:
1. Extracts entities, relationships, and key facts from claims
2. Creates targeted queries for entity clusters
3. Uses different k values for each hop (30, 20, 15)
4. Applies relevance-based reranking to select final 21 documents
"""

import dspy
from langProBe.hover.hover_program import HoverMultiHop, ClaimEntityExtractor
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline

def example_standalone_entity_extraction():
    """
    Example: Using ClaimEntityExtractor standalone to understand what it extracts.
    """
    print("=" * 80)
    print("EXAMPLE 1: Standalone Entity Extraction")
    print("=" * 80)

    # Configure DSPy (you'll need to set up your LM)
    # dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

    claim = "The director of The Matrix also directed Cloud Atlas with Tom Hanks."

    # Create the entity extractor
    extractor = dspy.ChainOfThought(ClaimEntityExtractor)

    # Extract entities
    print(f"\nClaim: {claim}\n")

    # Note: This will fail without proper DSPy configuration, but shows the interface
    try:
        extraction = extractor(claim=claim)
        print(f"Primary Entities: {extraction.primary_entities}")
        print(f"Secondary Entities: {extraction.secondary_entities}")
        print(f"Relationships: {extraction.relationships}")
        print(f"Key Facts: {extraction.key_facts}")
    except Exception as e:
        print(f"Note: Requires DSPy LM configuration. Error: {e}")


def example_full_pipeline():
    """
    Example: Using the full HoverMultiHopPipeline with entity-first retrieval.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Full Pipeline Execution")
    print("=" * 80)

    # Initialize the pipeline
    pipeline = HoverMultiHopPipeline()

    # Example claim
    claim = "Barack Obama was born in Hawaii and later became the 44th President of the United States."

    print(f"\nClaim: {claim}\n")
    print("Processing stages:")
    print("1. Extract entities: Barack Obama, Hawaii, 44th President, United States")
    print("2. Hop 1 (k=30): Retrieve docs about primary entities (Barack Obama, Hawaii)")
    print("3. Hop 2 (k=20): Retrieve docs about secondary entities (44th President, United States)")
    print("4. Hop 3 (k=15): Verify relationships (Obama born in Hawaii, Obama 44th President)")
    print("5. Rerank 65 docs and select top 21 with best entity coverage")

    # Run the pipeline
    try:
        result = pipeline(claim=claim)
        print(f"\n✓ Successfully retrieved {len(result.retrieved_docs)} documents")
        print(f"  First document: {result.retrieved_docs[0][:100]}...")
    except Exception as e:
        print(f"Note: Requires proper DSPy and retriever configuration. Error: {e}")


def example_multi_hop_breakdown():
    """
    Example: Detailed breakdown of the multi-hop retrieval process.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multi-Hop Retrieval Breakdown")
    print("=" * 80)

    claim = "Marie Curie won Nobel Prizes in both Physics and Chemistry."

    print(f"\nClaim: {claim}\n")

    print("PHASE 1: Entity Extraction")
    print("-" * 40)
    print("Primary Entities:")
    print("  • Marie Curie (person)")
    print("  • Nobel Prize (award)")
    print("Secondary Entities:")
    print("  • Physics (field)")
    print("  • Chemistry (field)")
    print("Relationships:")
    print("  • Marie Curie won Nobel Prize in Physics")
    print("  • Marie Curie won Nobel Prize in Chemistry")
    print("  • Same person won in two different fields")
    print("Key Facts:")
    print("  • Nobel Prize years")
    print("  • Specific achievements in each field")

    print("\nPHASE 2: Multi-Hop Targeted Retrieval")
    print("-" * 40)
    print("Hop 1 Query (k=30):")
    print("  'Marie Curie Nobel Prize biography'")
    print("  → Retrieves 30 docs about Marie Curie and Nobel Prize")

    print("\nHop 2 Query (k=20):")
    print("  'Marie Curie Physics Chemistry scientific work'")
    print("  → Retrieves 20 docs about her work in both fields")

    print("\nHop 3 Query (k=15):")
    print("  'Marie Curie won Nobel Prize Physics Chemistry both'")
    print("  → Retrieves 15 docs verifying she won in both fields")

    print("\nPHASE 3: Relevance-Based Reranking")
    print("-" * 40)
    print("Total documents: 65 (30 + 20 + 15)")
    print("Scoring criteria:")
    print("  • Mentions Marie Curie: +points")
    print("  • Mentions both Physics and Chemistry: +points")
    print("  • Confirms Nobel Prize awards: +points")
    print("  • Provides dates and details: +points")
    print("Result: Top 21 documents with best entity coverage")


def example_comparison_with_original():
    """
    Example: Comparison between original and new approach.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Original vs Entity-First Approach")
    print("=" * 80)

    claim = "The Eiffel Tower was built for the 1889 World's Fair in Paris."

    print(f"\nClaim: {claim}\n")

    print("ORIGINAL APPROACH:")
    print("-" * 40)
    print("Hop 1: Query='The Eiffel Tower was built for the 1889 World's Fair in Paris.'")
    print("       Retrieve k=7 documents")
    print("       Summarize results")
    print("\nHop 2: Generate query from claim + summary")
    print("       Retrieve k=7 documents")
    print("       Summarize results")
    print("\nHop 3: Generate query from claim + both summaries")
    print("       Retrieve k=7 documents")
    print("\nTotal: 21 documents (7+7+7), no reranking")
    print("Issue: May miss important entities (Paris, World's Fair, 1889)")

    print("\n\nNEW ENTITY-FIRST APPROACH:")
    print("-" * 40)
    print("Entity Extraction:")
    print("  Primary: Eiffel Tower, Paris")
    print("  Secondary: 1889, World's Fair")
    print("  Relationships: Eiffel Tower built for World's Fair")
    print("  Key Facts: 1889 date, Paris location")

    print("\nHop 1 (k=30): 'Eiffel Tower Paris history'")
    print("  → 30 docs about Eiffel Tower and Paris")

    print("\nHop 2 (k=20): '1889 World's Fair Paris exposition'")
    print("  → 20 docs about the World's Fair context")

    print("\nHop 3 (k=15): 'Eiffel Tower built 1889 World's Fair'")
    print("  → 15 docs verifying the specific relationship")

    print("\nReranking: Score all 65 docs → Select top 21")
    print("Result: Better entity coverage, higher quality documents")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HOVERMULTIHOP ENTITY-FIRST APPROACH - EXAMPLES")
    print("=" * 80)

    example_standalone_entity_extraction()
    example_full_pipeline()
    example_multi_hop_breakdown()
    example_comparison_with_original()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The new entity-extraction-first approach ensures:

✓ Explicit identification of all entities before retrieval
✓ Targeted queries for different entity types
✓ Comprehensive document pool (65 docs from 3 hops)
✓ Quality filtering via relevance-based reranking
✓ Final 21 documents optimized for entity coverage

Key improvements over original:
1. Entity awareness: Know what to look for before searching
2. Adaptive retrieval: Different k values for different entity types
3. Quality control: Reranking ensures best documents are selected
4. Coverage guarantee: Greedy selection maximizes entity coverage
    """)

    print("\nFor actual usage, configure DSPy with:")
    print("  dspy.configure(")
    print("      lm=dspy.LM('openai/gpt-4o-mini'),")
    print("      rm=dspy.ColBERTv2(url='your-colbert-server-url')")
    print("  )")
