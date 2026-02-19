"""Data structures for structured claim verification.

This module defines the core data structures used by the StructuredClaimVerifier
to represent verification results, evidence, contradictions, and sub-claim verdicts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any
from enum import Enum


class SubClaimVerdict(str, Enum):
    """Verdict for individual sub-claims."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNCLEAR = "UNCLEAR"


class OverallVerdict(str, Enum):
    """Final verdict for entire claim."""

    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass
class Evidence:
    """Single piece of evidence from retrieved documents.

    Attributes:
        document_id: Document title/key from which evidence was extracted
        quote: Exact quote from the document
        relevance_score: Confidence score (0-1) for relevance to sub-claim
        sentence_index: Position/index of the evidence in the document
    """

    document_id: str
    quote: str
    relevance_score: float
    sentence_index: int


@dataclass
class SubClaimVerification:
    """Verification result for a single atomic sub-claim.

    Attributes:
        sub_claim: The atomic sub-claim being verified
        verdict: Verification verdict (SUPPORTED/REFUTED/INSUFFICIENT/UNCLEAR)
        evidence: List of evidence pieces supporting or refuting the sub-claim
        confidence: Confidence score (0-1) in the verdict
        reasoning: Textual explanation of how evidence led to the verdict
    """

    sub_claim: str
    verdict: SubClaimVerdict
    evidence: List[Evidence]
    confidence: float
    reasoning: str


@dataclass
class Contradiction:
    """Detected logical inconsistency between sub-claims or evidence.

    Attributes:
        description: Human-readable description of the contradiction
        conflicting_subclaims: Indices of sub-claims involved in the contradiction
        conflicting_evidence: Evidence pieces that contradict each other
        severity: Severity level of the contradiction (high/medium/low)
    """

    description: str
    conflicting_subclaims: List[int]
    conflicting_evidence: List[Evidence]
    severity: Literal["high", "medium", "low"]


@dataclass
class VerificationResult:
    """Complete structured verification output.

    This is the main output of the StructuredClaimVerifier module, containing
    all verification information including sub-claims, evidence, contradictions,
    and the final verdict.

    Attributes:
        original_claim: The original claim that was verified
        sub_claims: List of all sub-claim verification results
        contradictions: List of detected logical contradictions
        overall_verdict: Final verdict for the entire claim
        confidence_score: Overall confidence in the final verdict (0-1)
        evidence_mapping: Mapping from sub-claim text to evidence list
        reasoning_chain: Step-by-step reasoning explaining the final verdict
        metadata: Additional metadata about the verification process
    """

    original_claim: str
    sub_claims: List[SubClaimVerification]
    contradictions: List[Contradiction]
    overall_verdict: OverallVerdict
    confidence_score: float
    evidence_mapping: Dict[str, List[Evidence]]
    reasoning_chain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
