#!/usr/bin/env python3
"""
Demonstration script for the StructuredClaimVerifier module.

This script shows how to use the StructuredClaimVerifier to:
1. Decompose claims into atomic sub-claims
2. Map evidence to each sub-claim
3. Detect logical contradictions
4. Generate structured verification results

Usage:
    python demo_verifier.py
"""

import dspy
from langProBe.hover.hover_verifier import StructuredClaimVerifier
from langProBe.hover.hover_program import HoverMultiHopWithVerification


def demo_structured_verifier():
    """Demonstrate the StructuredClaimVerifier with example data."""

    print("=" * 80)
    print("StructuredClaimVerifier Demonstration")
    print("=" * 80)
    print()

    # Initialize the verifier
    verifier = StructuredClaimVerifier(
        min_sub_claims=2,
        max_sub_claims=4,
        enable_consistency_check=True
    )

    print("✓ Initialized StructuredClaimVerifier")
    print(f"  - Min sub-claims: {verifier.min_sub_claims}")
    print(f"  - Max sub-claims: {verifier.max_sub_claims}")
    print(f"  - Consistency check: {verifier.enable_consistency_check}")
    print()

    # Example claim and passages
    claim = "The Austrian magazine was founded before the American magazine."
    passages = [
        "Magazine A | Magazine A is an American publication founded in 1990.",
        "Magazine B | Magazine B is an Austrian magazine established in 1985.",
        "Context | Magazine B, the Austrian publication, predates Magazine A by five years.",
        "History | Both magazines focus on current events and politics.",
    ]

    print("Example Claim:")
    print(f"  {claim}")
    print()

    print("Retrieved Passages:")
    for idx, passage in enumerate(passages, 1):
        doc_id, content = passage.split(" | ", 1)
        print(f"  [{idx}] {doc_id}: {content[:60]}...")
    print()

    # Note: This would require an LLM to be configured
    print("Note: To run the full verification, you would need to:")
    print("  1. Set up an LLM with: dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))")
    print("  2. Call: result = verifier(claim=claim, passages=passages)")
    print()

    print("Expected Output Structure:")
    print("  - original_claim: str")
    print("  - sub_claims: List[SubClaimVerification]")
    print("    - Each with: sub_claim, verdict, evidence, confidence, reasoning")
    print("  - contradictions: List[Contradiction]")
    print("  - overall_verdict: SUPPORTS|REFUTES|NOT_ENOUGH_INFO")
    print("  - confidence_score: float (0-1)")
    print("  - evidence_mapping: Dict[str, List[Evidence]]")
    print("  - reasoning_chain: str")
    print()

    # Test edge case handling without LLM
    print("Testing Edge Case Handling:")
    print()

    # Test 1: No passages
    result_empty = verifier._create_empty_result(claim)
    print(f"✓ Empty passages handled: verdict={result_empty.overall_verdict.value}")

    # Test 2: Validate sub-claims
    test_subclaims = ["Claim 1", "", "Claim 2", "Claim 3"]
    validated = verifier._validate_subclaims(test_subclaims, claim)
    print(f"✓ Sub-claim validation: {len(test_subclaims)} → {len(validated)} claims")

    # Test 3: Parse evidence
    test_quote = "Magazine A | Founded in 1990"
    evidence = verifier._parse_evidence(test_quote)
    print(f"✓ Evidence parsing: doc='{evidence.document_id}', quote='{evidence.quote[:30]}...'")

    print()
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


def demo_integrated_pipeline():
    """Demonstrate the integrated HoverMultiHopWithVerification pipeline."""

    print()
    print("=" * 80)
    print("Integrated Pipeline Demonstration")
    print("=" * 80)
    print()

    # Initialize the integrated pipeline
    pipeline = HoverMultiHopWithVerification(
        enable_verification=True,
        verifier_config={
            "min_sub_claims": 2,
            "max_sub_claims": 4,
            "enable_consistency_check": True
        }
    )

    print("✓ Initialized HoverMultiHopWithVerification pipeline")
    print("  - Retrieval: HoverMultiHop (3-hop, up to 21 docs)")
    print("  - Verification: StructuredClaimVerifier")
    print()

    print("Pipeline Flow:")
    print("  1. HoverMultiHop retrieves relevant documents (3 hops)")
    print("  2. StructuredClaimVerifier processes the claim:")
    print("     a. Decomposes into 2-4 atomic sub-claims")
    print("     b. Maps evidence to each sub-claim")
    print("     c. Checks for logical contradictions")
    print("     d. Aggregates into final verdict")
    print("  3. Returns: retrieved_docs, verification result, label")
    print()

    print("Note: To run the pipeline:")
    print("  1. Configure DSPy: dspy.configure(lm=..., rm=...)")
    print("  2. Call: result = pipeline(claim='...')")
    print("  3. Access: result.verification, result.retrieved_docs, result.label")
    print()

    print("=" * 80)


if __name__ == "__main__":
    demo_structured_verifier()
    demo_integrated_pipeline()

    print()
    print("For full examples with LLM calls, see:")
    print("  - tests/test_hover_verifier.py (unit tests)")
    print("  - tests/test_pipelines.py (integration tests)")
    print()
