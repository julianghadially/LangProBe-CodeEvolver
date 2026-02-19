"""Structured claim verification module.

This module implements the StructuredClaimVerifier, which operates after retrieval
and before final answer generation. It decomposes claims into atomic sub-claims,
maps evidence explicitly, performs logical consistency checking, and outputs
structured verification results.

Pipeline:
1. Decompose claim into 2-4 atomic sub-claims
2. Map evidence to each sub-claim with specific quotes
3. Check logical consistency across sub-claims
4. Aggregate into final verdict with evidence mapping
"""

import dspy
import re
from typing import List, Dict, Tuple
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_verifier_signatures import (
    ClaimDecompositionSignature,
    EvidenceMappingSignature,
    ConsistencyCheckSignature,
    FinalVerdictSignature,
)
from .hover_verifier_models import (
    Evidence,
    SubClaimVerification,
    Contradiction,
    VerificationResult,
    SubClaimVerdict,
    OverallVerdict,
)


class StructuredClaimVerifier(LangProBeDSPyMetaProgram, dspy.Module):
    """Structured claim verification module that operates after retrieval.

    This module takes a claim and retrieved documents, then performs fine-grained
    verification including:
    - Decomposition into atomic sub-claims
    - Evidence mapping with specific quotes
    - Logical consistency checking
    - Structured output with evidence attribution

    Usage:
        verifier = StructuredClaimVerifier(
            min_sub_claims=2,
            max_sub_claims=4,
            contradiction_threshold=0.3
        )
        result = verifier(claim="...", passages=[...])

    Args:
        min_sub_claims: Minimum number of sub-claims to generate (default: 2)
        max_sub_claims: Maximum number of sub-claims to generate (default: 4)
        contradiction_threshold: Threshold for contradiction severity (default: 0.3)
        enable_consistency_check: Whether to perform consistency checking (default: True)
    """

    def __init__(
        self,
        min_sub_claims: int = 2,
        max_sub_claims: int = 4,
        contradiction_threshold: float = 0.3,
        enable_consistency_check: bool = True,
    ):
        super().__init__()
        self.min_sub_claims = min_sub_claims
        self.max_sub_claims = max_sub_claims
        self.contradiction_threshold = contradiction_threshold
        self.enable_consistency_check = enable_consistency_check

        # Initialize sub-modules using ChainOfThought
        self.decompose = dspy.ChainOfThought(ClaimDecompositionSignature)
        self.map_evidence = dspy.ChainOfThought(EvidenceMappingSignature)
        self.check_consistency = dspy.ChainOfThought(ConsistencyCheckSignature)
        self.aggregate_verdict = dspy.ChainOfThought(FinalVerdictSignature)

    def forward(self, claim: str, passages: List[str]) -> VerificationResult:
        """Main forward pass for structured verification.

        Args:
            claim: The claim to verify
            passages: List of retrieved documents (format: "Title | Content")

        Returns:
            VerificationResult with complete structured output including:
            - Sub-claims with individual verdicts
            - Evidence mapping
            - Contradictions detected
            - Overall verdict and confidence
        """
        # Edge case: no passages provided
        if not passages:
            return self._create_empty_result(claim)

        # Step 1: Decompose claim into sub-claims
        decomposition = self._decompose_claim(claim, passages)
        sub_claims = self._validate_subclaims(decomposition["sub_claims"], claim)

        # Step 2: Map evidence for each sub-claim
        sub_claim_verifications = []
        evidence_mapping = {}

        for sub_claim in sub_claims:
            verification = self._verify_subclaim(sub_claim, passages)
            sub_claim_verifications.append(verification)
            evidence_mapping[sub_claim] = verification.evidence

        # Step 3: Check consistency (optional)
        contradictions = []
        if self.enable_consistency_check and len(sub_claim_verifications) > 1:
            contradictions = self._check_consistency(claim, sub_claim_verifications)

        # Step 4: Aggregate final verdict
        overall_verdict, confidence, reasoning = self._aggregate_verdict(
            claim, sub_claim_verifications, contradictions
        )

        return VerificationResult(
            original_claim=claim,
            sub_claims=sub_claim_verifications,
            contradictions=contradictions,
            overall_verdict=overall_verdict,
            confidence_score=confidence,
            evidence_mapping=evidence_mapping,
            reasoning_chain=reasoning,
            metadata={
                "num_passages": len(passages),
                "num_sub_claims": len(sub_claims),
                "consistency_check_enabled": self.enable_consistency_check,
            },
        )

    def _create_empty_result(self, claim: str) -> VerificationResult:
        """Create a result for cases with no passages."""
        return VerificationResult(
            original_claim=claim,
            sub_claims=[],
            contradictions=[],
            overall_verdict=OverallVerdict.NOT_ENOUGH_INFO,
            confidence_score=0.0,
            evidence_mapping={},
            reasoning_chain="No passages provided for verification.",
            metadata={"num_passages": 0, "num_sub_claims": 0},
        )

    def _decompose_claim(self, claim: str, passages: List[str]) -> Dict:
        """Step 1: Decompose claim into atomic sub-claims.

        Args:
            claim: The claim to decompose
            passages: Retrieved documents for context

        Returns:
            Dictionary with 'sub_claims' (list) and 'reasoning' (str)
        """
        # Use first 5 passages for context to avoid token limits
        context_passages = "\n".join(passages[:5])

        pred = self.decompose(claim=claim, passages=context_passages)

        # Parse sub_claims from output
        sub_claims = pred.sub_claims
        if isinstance(sub_claims, str):
            # Handle case where LLM returns string instead of list
            sub_claims = self._parse_list_output(sub_claims)

        return {"sub_claims": sub_claims, "reasoning": pred.decomposition_reasoning}

    def _validate_subclaims(self, sub_claims: List[str], original_claim: str) -> List[str]:
        """Validate and normalize sub-claims list.

        Args:
            sub_claims: List of sub-claims from decomposition
            original_claim: Original claim for fallback

        Returns:
            Validated list of sub-claims
        """
        # Filter out empty sub-claims
        sub_claims = [sc.strip() for sc in sub_claims if sc and sc.strip()]

        # Handle too few sub-claims: fall back to original claim
        if len(sub_claims) < self.min_sub_claims:
            return [original_claim]

        # Handle too many sub-claims: truncate
        if len(sub_claims) > self.max_sub_claims:
            sub_claims = sub_claims[: self.max_sub_claims]

        return sub_claims

    def _verify_subclaim(
        self, sub_claim: str, passages: List[str]
    ) -> SubClaimVerification:
        """Step 2: Map evidence and verify a single sub-claim.

        Args:
            sub_claim: The atomic sub-claim to verify
            passages: All retrieved documents

        Returns:
            SubClaimVerification with verdict, evidence, confidence, reasoning
        """
        # Format passages for input
        passages_str = "\n\n".join(passages)

        pred = self.map_evidence(sub_claim=sub_claim, passages=passages_str)

        # Parse evidence quotes
        evidence_list = []
        quotes = pred.relevant_quotes
        if isinstance(quotes, str):
            quotes = self._parse_list_output(quotes)

        for quote_str in quotes:
            evidence = self._parse_evidence(quote_str)
            if evidence:
                evidence_list.append(evidence)

        # Parse verdict and confidence
        verdict = self._parse_verdict(pred.verdict)
        confidence = self._parse_confidence(pred.confidence)

        return SubClaimVerification(
            sub_claim=sub_claim,
            verdict=verdict,
            evidence=evidence_list,
            confidence=confidence,
            reasoning=pred.reasoning,
        )

    def _check_consistency(
        self, claim: str, verifications: List[SubClaimVerification]
    ) -> List[Contradiction]:
        """Step 3: Check for logical contradictions.

        Args:
            claim: Original claim
            verifications: List of sub-claim verifications

        Returns:
            List of detected contradictions
        """
        # Format sub-claim verdicts for input
        verdicts_str = self._format_subclaim_verdicts(verifications)

        pred = self.check_consistency(
            original_claim=claim, sub_claim_verdicts=verdicts_str
        )

        # Parse contradictions
        contradictions = []
        contradiction_strs = pred.contradictions
        if isinstance(contradiction_strs, str):
            contradiction_strs = self._parse_list_output(contradiction_strs)

        for contra_str in contradiction_strs:
            contra = self._parse_contradiction(contra_str, verifications)
            if contra:
                contradictions.append(contra)

        return contradictions

    def _aggregate_verdict(
        self,
        claim: str,
        verifications: List[SubClaimVerification],
        contradictions: List[Contradiction],
    ) -> Tuple[OverallVerdict, float, str]:
        """Step 4: Aggregate sub-claim verdicts into final verdict.

        Args:
            claim: Original claim
            verifications: All sub-claim verifications
            contradictions: Detected contradictions

        Returns:
            Tuple of (overall_verdict, confidence_score, reasoning_chain)
        """
        sub_results_str = self._format_subclaim_verdicts(verifications)
        contradictions_str = self._format_contradictions(contradictions)

        pred = self.aggregate_verdict(
            original_claim=claim,
            sub_claim_results=sub_results_str,
            contradictions=contradictions_str,
        )

        overall_verdict = self._parse_overall_verdict(pred.overall_verdict)
        confidence = self._parse_confidence(pred.confidence_score)

        return overall_verdict, confidence, pred.reasoning_chain

    # ==================== Parsing Helper Methods ====================

    def _parse_list_output(self, text: str) -> List[str]:
        """Parse LLM list output that may be formatted various ways.

        Handles:
        - Numbered lists: "1. item\n2. item"
        - Bulleted lists: "- item\n* item"
        - Newline-separated items

        Args:
            text: Raw text output from LLM

        Returns:
            List of parsed items
        """
        # Try numbered list: "1. ...\n2. ..."
        numbered = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL)
        if numbered:
            return [item.strip() for item in numbered if item.strip()]

        # Try bullet list: "- ...\n- ..." or "* ...\n* ..."
        bulleted = re.findall(r"[-*]\s*(.+?)(?=\n[-*]|\Z)", text, re.DOTALL)
        if bulleted:
            return [item.strip() for item in bulleted if item.strip()]

        # Fall back to newline split
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter out lines that look like headers or separators
        items = [
            line
            for line in lines
            if len(line) > 3 and not line.startswith("---") and not line.startswith("===")
        ]
        return items if items else [text.strip()]

    def _parse_evidence(self, quote_str: str) -> Evidence:
        """Parse evidence from 'Title | Quote' format.

        Args:
            quote_str: String in format "DocumentTitle | Quote text"

        Returns:
            Evidence object or None if parsing fails
        """
        # Split on first pipe character
        parts = quote_str.split("|", 1)
        if len(parts) < 2:
            # If no pipe, treat entire string as quote with unknown document
            return Evidence(
                document_id="Unknown",
                quote=quote_str.strip(),
                relevance_score=0.5,
                sentence_index=0,
            )

        return Evidence(
            document_id=parts[0].strip(),
            quote=parts[1].strip(),
            relevance_score=0.8,  # Default relevance score
            sentence_index=0,  # Default position
        )

    def _parse_verdict(self, verdict_str: str) -> SubClaimVerdict:
        """Parse verdict from string to enum.

        Args:
            verdict_str: Verdict string from LLM output

        Returns:
            SubClaimVerdict enum value
        """
        verdict_upper = str(verdict_str).upper().strip()

        if "SUPPORT" in verdict_upper and "NOT" not in verdict_upper:
            return SubClaimVerdict.SUPPORTED
        elif "REFUT" in verdict_upper:
            return SubClaimVerdict.REFUTED
        elif "INSUFFICIENT" in verdict_upper or "NOT_ENOUGH" in verdict_upper:
            return SubClaimVerdict.INSUFFICIENT
        else:
            return SubClaimVerdict.UNCLEAR

    def _parse_overall_verdict(self, verdict_str: str) -> OverallVerdict:
        """Parse overall verdict from string to enum.

        Args:
            verdict_str: Verdict string from LLM output

        Returns:
            OverallVerdict enum value
        """
        verdict_upper = str(verdict_str).upper().strip()

        if "SUPPORT" in verdict_upper and "NOT" not in verdict_upper:
            return OverallVerdict.SUPPORTS
        elif "REFUT" in verdict_upper:
            return OverallVerdict.REFUTES
        else:
            return OverallVerdict.NOT_ENOUGH_INFO

    def _parse_confidence(self, confidence_str) -> float:
        """Parse confidence score, handling various formats.

        Handles:
        - Float values: 0.85
        - Percentage strings: "85%"
        - Integer percentages: "85"

        Args:
            confidence_str: Confidence value in various formats

        Returns:
            Float between 0.0 and 1.0
        """
        if isinstance(confidence_str, (int, float)):
            val = float(confidence_str)
            # Normalize to 0-1 if needed
            return val if val <= 1.0 else val / 100.0

        # Extract first number from string
        match = re.search(r"(\d+\.?\d*)", str(confidence_str))
        if match:
            val = float(match.group(1))
            # Normalize to 0-1 if needed
            return val if val <= 1.0 else val / 100.0

        # Default to medium confidence if parsing fails
        return 0.5

    def _parse_contradiction(
        self, contra_str: str, verifications: List[SubClaimVerification]
    ) -> Contradiction:
        """Parse contradiction from formatted string.

        Args:
            contra_str: Contradiction string from LLM output
            verifications: List of sub-claim verifications for context

        Returns:
            Contradiction object or None if parsing fails
        """
        # Extract severity
        severity = "medium"
        contra_lower = contra_str.lower()
        if "severity: high" in contra_lower or "high severity" in contra_lower:
            severity = "high"
        elif "severity: low" in contra_lower or "low severity" in contra_lower:
            severity = "low"

        # Extract sub-claim indices (look for [0], [1], etc.)
        indices = re.findall(r"\[(\d+)\]", contra_str)
        conflicting_indices = [int(i) for i in indices]

        # Collect evidence from conflicting sub-claims
        conflicting_evidence = []
        for idx in conflicting_indices:
            if 0 <= idx < len(verifications):
                conflicting_evidence.extend(verifications[idx].evidence)

        return Contradiction(
            description=contra_str,
            conflicting_subclaims=conflicting_indices,
            conflicting_evidence=conflicting_evidence,
            severity=severity,
        )

    # ==================== Formatting Helper Methods ====================

    def _format_subclaim_verdicts(
        self, verifications: List[SubClaimVerification]
    ) -> str:
        """Format sub-claim verifications for LLM input.

        Args:
            verifications: List of sub-claim verifications

        Returns:
            Formatted string representation
        """
        lines = []
        for idx, v in enumerate(verifications):
            evidence_summary = f"{len(v.evidence)} quote(s)"
            if v.evidence:
                # Include first quote as example
                first_quote = v.evidence[0].quote[:100]
                if len(v.evidence[0].quote) > 100:
                    first_quote += "..."
                evidence_summary += f' (e.g., "{first_quote}")'

            lines.append(
                f"[{idx}] Sub-claim: {v.sub_claim}\n"
                f"    Verdict: {v.verdict.value}\n"
                f"    Confidence: {v.confidence:.2f}\n"
                f"    Evidence: {evidence_summary}\n"
                f"    Reasoning: {v.reasoning}"
            )
        return "\n\n".join(lines)

    def _format_contradictions(self, contradictions: List[Contradiction]) -> str:
        """Format contradictions for LLM input.

        Args:
            contradictions: List of contradictions

        Returns:
            Formatted string representation
        """
        if not contradictions:
            return "No contradictions detected"

        lines = []
        for idx, c in enumerate(contradictions):
            conflicting_indices = ", ".join(str(i) for i in c.conflicting_subclaims)
            lines.append(
                f"Contradiction {idx+1}: {c.description}\n"
                f"    Affects sub-claims: [{conflicting_indices}]\n"
                f"    Severity: {c.severity}"
            )
        return "\n\n".join(lines)
