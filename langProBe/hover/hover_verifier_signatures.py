"""DSPy signatures for structured claim verification.

This module defines the DSPy signatures used by the StructuredClaimVerifier
for each step of the verification pipeline:
1. Claim decomposition
2. Evidence mapping
3. Consistency checking
4. Verdict aggregation
"""

import dspy
from typing import List


class ClaimDecompositionSignature(dspy.Signature):
    """Decompose a complex claim into 2-4 atomic, independently verifiable sub-claims.

    Each sub-claim should:
    - Be a simple factual statement
    - Be independently verifiable
    - Contain no logical connectives (and, or, but)
    - Preserve the semantic meaning of the original claim

    Example:
        Input: "The Austrian magazine was founded before the American magazine"
        Output: ["Magazine A is Austrian", "Magazine B is American",
                 "Magazine A was founded before Magazine B"]
    """

    claim = dspy.InputField(desc="The original claim to decompose")
    passages = dspy.InputField(desc="Retrieved documents providing context")

    sub_claims: List[str] = dspy.OutputField(
        desc="List of 2-4 atomic sub-claims. Each sub-claim must be a complete sentence."
    )
    decomposition_reasoning = dspy.OutputField(
        desc="Brief explanation of how the claim was decomposed"
    )


class EvidenceMappingSignature(dspy.Signature):
    """Find and extract specific evidence from documents to verify a sub-claim.

    For each relevant passage, extract:
    - The exact quote supporting or refuting the sub-claim
    - The document source
    - A relevance assessment

    The verdict should be:
    - SUPPORTED: If evidence clearly supports the sub-claim
    - REFUTED: If evidence clearly contradicts the sub-claim
    - INSUFFICIENT: If not enough evidence exists
    - UNCLEAR: If evidence is ambiguous or contradictory

    Example:
        Input sub_claim: "Magazine A is American"
        Input passages: ["Magazine A | Magazine A is an American publication..."]
        Output: quotes=["Magazine A | Magazine A is an American publication"],
                verdict="SUPPORTED", confidence=0.95
    """

    sub_claim = dspy.InputField(desc="The atomic sub-claim to verify")
    passages = dspy.InputField(
        desc="Retrieved documents in format 'Title | Content'"
    )

    relevant_quotes: List[str] = dspy.OutputField(
        desc="Exact quotes from passages (with document titles). Format: 'Title | Quote'"
    )
    verdict = dspy.OutputField(
        desc="Verdict for this sub-claim: SUPPORTED, REFUTED, INSUFFICIENT, or UNCLEAR"
    )
    confidence = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")
    reasoning = dspy.OutputField(
        desc="Explanation of why the evidence leads to this verdict"
    )


class ConsistencyCheckSignature(dspy.Signature):
    """Identify logical contradictions between sub-claims and evidence.

    Check for:
    - Direct contradictions (claim says X, evidence says not-X)
    - Indirect contradictions (sub-claim A and sub-claim B cannot both be true)
    - Temporal inconsistencies (dates don't align)
    - Entity mismatches (same entity described with contradicting attributes)

    Example contradictions to detect:
    - "Magazine A is Austrian" vs "Magazine A is American"
    - "Founded in 1990" vs "Founded in 1995"
    - "Person was alive in 2000" vs "Person died in 1995"

    For each contradiction, format as:
    "Sub-claim [index]: Description of contradiction | Severity: high/medium/low"
    """

    original_claim = dspy.InputField(desc="The original claim")
    sub_claim_verdicts = dspy.InputField(
        desc="List of sub-claims with their verdicts and evidence"
    )

    contradictions: List[str] = dspy.OutputField(
        desc="List of contradictions found. Format: 'Sub-claim [index]: Description | Severity: high/medium/low'"
    )
    consistency_score = dspy.OutputField(
        desc="Overall consistency score from 0.0 (many contradictions) to 1.0 (fully consistent)"
    )


class FinalVerdictSignature(dspy.Signature):
    """Aggregate sub-claim verdicts into final overall verdict for the claim.

    Consider:
    - How many sub-claims are supported vs refuted
    - Confidence scores of each sub-claim
    - Severity of any contradictions
    - Whether all sub-claims must be true for the claim to be supported
    - Logical dependencies between sub-claims

    The final verdict should be:
    - SUPPORTS: If the evidence supports the overall claim
    - REFUTES: If the evidence refutes the overall claim
    - NOT_ENOUGH_INFO: If there is insufficient evidence to make a determination

    Example:
        If claim requires A AND B to be true:
        - If both A and B are SUPPORTED → SUPPORTS
        - If either A or B is REFUTED → REFUTES
        - If evidence is insufficient → NOT_ENOUGH_INFO
    """

    original_claim = dspy.InputField(desc="The original claim to verify")
    sub_claim_results = dspy.InputField(
        desc="All sub-claim verification results with verdicts and confidence"
    )
    contradictions = dspy.InputField(desc="List of detected contradictions")

    overall_verdict = dspy.OutputField(
        desc="Final verdict: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO"
    )
    confidence_score = dspy.OutputField(
        desc="Overall confidence from 0.0 to 1.0"
    )
    reasoning_chain = dspy.OutputField(
        desc="Step-by-step reasoning explaining how sub-claim verdicts lead to the final verdict"
    )
