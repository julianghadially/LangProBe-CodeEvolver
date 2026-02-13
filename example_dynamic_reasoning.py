#!/usr/bin/env python3
"""
Example demonstrating the dynamic sequential reasoning architecture.

This script shows how the new HoverMultiHopPredict discovers implicit entities
through retrieved document content, enabling true multi-hop reasoning chains.
"""

import dspy
from langProBe.hover.hover_program import HoverMultiHopPredict

def demonstrate_architecture():
    """Demonstrate the new dynamic reasoning flow."""
    print("=" * 80)
    print("Dynamic Sequential Reasoning Architecture for HoverMultiHopPredict")
    print("=" * 80)

    print("\n📋 OVERVIEW")
    print("-" * 80)
    print("The new architecture replaces static entity extraction with dynamic")
    print("sequential reasoning that:")
    print("  1. Analyzes the claim to determine initial information needs")
    print("  2. Summarizes retrieved documents to extract key facts")
    print("  3. Identifies information gaps and implicit entities")
    print("  4. Generates targeted queries based on discovered information")

    print("\n🔍 EXAMPLE CLAIM")
    print("-" * 80)
    example_claim = "Lisa Raymond's partner won the 1999 French Open doubles title"
    print(f"Claim: \"{example_claim}\"")

    print("\n🎯 EXECUTION FLOW")
    print("-" * 80)

    print("\n[Hop 1] FirstHopPlanner")
    print("  Input: claim only")
    print("  Reasoning: 'Need to identify who Lisa Raymond's partner was in 1999")
    print("             French Open doubles'")
    print("  Query: '1999 French Open women's doubles champions'")
    print("  Retrieved: Documents revealing Lisa Raymond won with Martina Hingis ✨")

    print("\n[Hop 2] NextHopPlanner")
    print("  Input: claim + previous_queries + retrieved_titles + key_facts")
    print("  Key Facts: 'Lisa Raymond won 1999 French Open with Martina Hingis'")
    print("  Information Gap: 'Need to verify Martina Hingis Grand Slam record'")
    print("  Reasoning: 'Martina Hingis is the implicit entity discovered from docs'")
    print("  Query: 'Martina Hingis Grand Slam singles doubles titles'")
    print("  Retrieved: Documents about Martina Hingis's career ✨")

    print("\n[Hop 3] NextHopPlanner")
    print("  Input: claim + previous_queries + retrieved_titles + key_facts")
    print("  Key Facts: 'Martina Hingis has 5 Grand Slam singles, 13 doubles titles'")
    print("  Information Gap: 'Need to confirm 1999 French Open doubles victory'")
    print("  Query: 'Martina Hingis 1999 French Open doubles final'")
    print("  Retrieved: Confirmation documents")

    print("\n✨ KEY INNOVATION")
    print("-" * 80)
    print("The system discovered 'Martina Hingis' (NOT mentioned in the original claim)")
    print("through document content, enabling true multi-hop reasoning chains!")

    print("\n📊 ARCHITECTURE COMPARISON")
    print("-" * 80)
    print("OLD APPROACH (Static Entity Extraction):")
    print("  1. Extract 3-4 entities upfront from claim only")
    print("  2. For each hop, select which pre-extracted entity to search")
    print("  3. Cannot discover implicit entities from documents")
    print()
    print("NEW APPROACH (Dynamic Sequential Reasoning):")
    print("  1. Analyze claim to determine initial information needs")
    print("  2. After each retrieval, summarize key facts from documents")
    print("  3. Identify information gaps and implicit entities")
    print("  4. Generate next query based on discovered information")

    print("\n🔧 TECHNICAL DETAILS")
    print("-" * 80)
    print("New Signatures:")
    print("  • FirstHopPlanner(claim) → reasoning, search_query")
    print("  • NextHopPlanner(claim, previous_queries, retrieved_titles,")
    print("                   key_facts_found) → information_gap, reasoning,")
    print("                                      search_query")
    print()
    print("New Helper Methods:")
    print("  • _summarize_documents(docs) → Extracts title + first line of content")
    print("    to capture key entities and facts efficiently")
    print()
    print("Context Accumulation:")
    print("  • previous_queries: List of all executed queries")
    print("  • retrieved_titles: Set of all retrieved document titles")
    print("  • key_facts_found: Summary of most recent hop's documents")

    print("\n🎨 DESIGN PATTERNS")
    print("-" * 80)
    print("Follows DSPy Best Practices:")
    print("  • ChainOfThought wrapping for reasoning tasks")
    print("  • Multi-input signatures with descriptive field descriptions")
    print("  • Context accumulation across hops (SimplifiedBaleen pattern)")
    print("  • Document deduplication by title")
    print("  • Maintains same interface (backward compatible)")

    print("\n🧪 TESTING")
    print("-" * 80)
    print("Run basic tests:")
    print("  python test_dynamic_reasoning.py")
    print()
    print("Test with actual retrieval (requires LM and retriever setup):")
    print("  from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline")
    print("  import dspy")
    print("  lm = dspy.LM('openai/gpt-4o-mini')  # or your preferred model")
    print("  dspy.settings.configure(lm=lm)")
    print("  pipeline = HoverMultiHopPredictPipeline()")
    print("  result = pipeline(claim='Lisa Raymond\\'s partner won...')")
    print("  print(f'Retrieved {len(result.retrieved_docs)} documents')")

    print("\n" + "=" * 80)
    print("Implementation complete! ✓")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_architecture()
