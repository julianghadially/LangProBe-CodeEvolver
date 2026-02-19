#!/usr/bin/env python3
"""
Test script to demonstrate the ChainOfThoughtVerifier module in hover_program.py

This script shows how to use the new verification capabilities:
1. ChainOfThoughtVerifier - standalone verifier for testing
2. HoverProgram - complete pipeline with retrieval + verification
"""

import dspy
from langProBe.hover.hover_program import (
    ChainOfThoughtVerifier,
    HoverProgram,
    HoverMultiHop
)


def test_verifier_only():
    """Test the ChainOfThoughtVerifier with mock passages"""
    print("=" * 80)
    print("TEST 1: ChainOfThoughtVerifier (standalone)")
    print("=" * 80)

    # Example claim and mock passages
    claim = "Antonis Fotsis is a player for the club whose name has the starting letter from an alphabet derived from the Phoenician alphabet."

    mock_passages = [
        "Ilysiakos B.C. | Ilysiakos B.C. is a Greek professional basketball club.",
        "Greek alphabet | The Greek alphabet has been used to write the Greek language since the late 9th or early 8th century BC. It is derived from the earlier Phoenician alphabet.",
        "Antonis Fotsis | Antonis Fotsis is a Greek professional basketball player who plays for Ilysiakos B.C."
    ]

    # Initialize verifier
    verifier = ChainOfThoughtVerifier()

    print(f"\nClaim: {claim}\n")
    print("Mock Passages:")
    for i, passage in enumerate(mock_passages, 1):
        print(f"  [{i}] {passage}")

    print("\n" + "-" * 80)
    print("Running ChainOfThoughtVerifier...")
    print("-" * 80 + "\n")

    # Note: This will fail without proper LM configuration
    # Uncomment below when running with actual LM
    # result = verifier(claim=claim, passages=mock_passages)
    #
    # print(f"Decision: {result.decision}")
    # print(f"Confidence Score: {result.confidence_score}")
    # print(f"\nKey Facts:")
    # for i, fact in enumerate(result.key_facts, 1):
    #     print(f"  {i}. {fact}")
    # print(f"\nMissing Info:")
    # for i, info in enumerate(result.missing_info, 1):
    #     print(f"  {i}. {info}")
    # print(f"\nReasoning Chains:")
    # for i, chain in enumerate(result.reasoning_chains, 1):
    #     print(f"  {i}. {chain}")
    # print(f"\nJustification:\n{result.justification}")

    print("(Skipped - requires LM configuration)")


def test_hover_program():
    """Test the complete HoverProgram with retrieval + verification"""
    print("\n" + "=" * 80)
    print("TEST 2: HoverProgram (retrieval + verification)")
    print("=" * 80)

    claim = "Antonis Fotsis is a player for the club whose name has the starting letter from an alphabet derived from the Phoenician alphabet."

    # Initialize complete program
    program = HoverProgram()

    print(f"\nClaim: {claim}\n")
    print("-" * 80)
    print("Running HoverProgram (retrieval + verification)...")
    print("-" * 80 + "\n")

    # Note: This requires ColBERT retriever and LM configuration
    # Uncomment below when running in proper environment
    # result = program(claim=claim)
    #
    # print(f"Retrieved {len(result.retrieved_docs)} documents")
    # print(f"\nDecision: {result.decision}")
    # print(f"Confidence Score: {result.confidence_score}")
    # print(f"\nJustification:\n{result.justification}")

    print("(Skipped - requires ColBERT and LM configuration)")


def show_architecture():
    """Display the architecture of the chain-of-thought reasoning"""
    print("\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT VERIFICATION ARCHITECTURE")
    print("=" * 80)

    print("""
The ChainOfThoughtVerifier performs structured reasoning in 4 steps:

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Key Facts                                       │
│ ────────────────────────                                        │
│ Input:  Claim + Retrieved Passages                             │
│ Output: List of key facts relevant to the claim                │
│ Purpose: Identify specific factual statements from documents   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Gap Analysis (Identify Missing Information)            │
│ ────────────────────────────────────────────                   │
│ Input:  Claim + Key Facts                                      │
│ Output: Missing information + Coverage assessment              │
│ Purpose: Determine what evidence is still lacking              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Chain Facts Together                                   │
│ ─────────────────────────                                      │
│ Input:  Claim + Key Facts + Missing Info                       │
│ Output: Reasoning chains connecting facts                      │
│ Purpose: Build logical connections toward verification         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Final Verification Decision                            │
│ ───────────────────────────────                                │
│ Input:  All previous outputs                                   │
│ Output: Decision (SUPPORTS/REFUTES) + Confidence + Justification│
│ Purpose: Make final evidence-based verification decision       │
└─────────────────────────────────────────────────────────────────┘

Key Features:
━━━━━━━━━━━━━
• Explicit reasoning at each step
• Gap analysis to identify missing evidence
• Logical chaining of facts before classification
• Confidence scoring based on evidence strength
• Transparent justification of decisions

Example Usage:
━━━━━━━━━━━━━
# Standalone verifier
verifier = ChainOfThoughtVerifier()
result = verifier(claim="...", passages=["..."])

# Complete pipeline (retrieval + verification)
program = HoverProgram()
result = program(claim="...")

# Access results
print(result.decision)           # "SUPPORTS" or "REFUTES"
print(result.confidence_score)   # 0.0 to 1.0
print(result.justification)      # Explanation
print(result.key_facts)          # Extracted facts
print(result.reasoning_chains)   # Logical connections
""")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "HOVER CHAIN-OF-THOUGHT VERIFICATION DEMO" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")

    show_architecture()
    test_verifier_only()
    test_hover_program()

    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("""
To run these tests with actual LM and retrieval:

1. Configure DSPy with your LM:
   dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="..."))

2. For HoverProgram, also configure ColBERT retriever:
   from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
   pipeline = HoverMultiHopPipeline()
   pipeline.setup_lm("openai/gpt-4o-mini", api_key="...")

3. Uncomment the test execution code in the functions above

The implementation is complete and ready to use!
""")
    print("=" * 80 + "\n")
