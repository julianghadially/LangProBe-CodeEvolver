#!/usr/bin/env python3
"""
Test script for the new Serper + Firecrawl HotPot implementation.

This script demonstrates the new two-stage search architecture:
1. Serper searches for relevant Wikipedia pages
2. Firecrawl scrapes the top result from each search
3. Full page content is used for context instead of abstracts
"""

import dspy
from langProPlus.hotpotGEPA.hotpot_program import HotpotMultiHopPredict

def test_simple_question():
    """Test with a simple multi-hop question."""

    # Initialize the program
    program = HotpotMultiHopPredict()

    # Example HotPot QA question
    question = "What is the nationality of the person who directed the film that won the Academy Award for Best Picture in 2020?"

    print(f"Question: {question}\n")
    print("=" * 80)
    print("Starting multi-hop reasoning with Serper + Firecrawl...\n")

    try:
        result = program(question=question)
        print(f"\nFinal Answer: {result.answer}")
        print("=" * 80)
        return result.answer
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Note: This requires:
    # - SERPER_KEY environment variable set
    # - FIRECRAWL_KEY environment variable set
    # - dspy configured with an LM

    print("HotPot Multi-Hop QA with Serper + Firecrawl")
    print("=" * 80)
    print("\nNOTE: Before running, ensure you have:")
    print("  1. Set SERPER_KEY environment variable")
    print("  2. Set FIRECRAWL_KEY environment variable")
    print("  3. Configured dspy with an LM (e.g., dspy.OpenAI)")
    print("\n" + "=" * 80 + "\n")

    test_simple_question()
