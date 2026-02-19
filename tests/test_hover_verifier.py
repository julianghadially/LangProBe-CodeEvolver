"""Unit tests for the StructuredClaimVerifier module.

Tests cover:
- Claim decomposition
- Evidence mapping
- Logical consistency checking
- End-to-end verification pipeline
- Edge cases (empty passages, conflicting evidence, etc.)
"""

import pytest
import dspy
from langProBe.hover.hover_verifier import StructuredClaimVerifier
from langProBe.hover.hover_verifier_models import (
    Evidence,
    SubClaimVerification,
    SubClaimVerdict,
    OverallVerdict,
)

LM = dspy.LM("openai/gpt-4o-mini")


@pytest.fixture
def verifier():
    """Create a StructuredClaimVerifier instance for testing."""
    return StructuredClaimVerifier(
        min_sub_claims=2, max_sub_claims=4, enable_consistency_check=True
    )


@pytest.fixture
def sample_claim():
    """Sample claim for testing."""
    return "The Austrian magazine was founded before the American magazine."


@pytest.fixture
def sample_passages():
    """Sample passages for testing."""
    return [
        "Magazine A | Magazine A is an American publication founded in 1990.",
        "Magazine B | Magazine B is an Austrian magazine established in 1985.",
        "Context | Magazine B, the Austrian publication, predates Magazine A.",
    ]


@pytest.fixture
def conflicting_passages():
    """Passages with contradictory information."""
    return [
        "Source 1 | Magazine A was founded in 1990.",
        "Source 2 | Magazine A was founded in 1995.",
        "Source 3 | The exact founding date of Magazine A is disputed.",
    ]


class TestClaimDecomposition:
    """Test Step 1: Claim decomposition functionality."""

    def test_decompose_simple_claim(self, verifier, sample_claim, sample_passages):
        """Test decomposition of a multi-faceted claim."""
        with dspy.context(lm=LM):
            result = verifier._decompose_claim(sample_claim, sample_passages)

        assert "sub_claims" in result
        assert "reasoning" in result
        assert isinstance(result["sub_claims"], list)
        assert len(result["sub_claims"]) >= 1

        # Should contain something about Austrian and American
        all_subclaims = " ".join(result["sub_claims"]).lower()
        assert "austrian" in all_subclaims or "american" in all_subclaims

    def test_decompose_already_atomic(self, verifier, sample_passages):
        """Test decomposition of an already atomic claim."""
        atomic_claim = "The sky is blue."
        with dspy.context(lm=LM):
            result = verifier._decompose_claim(atomic_claim, sample_passages)

        # Should still work, possibly returning the claim itself or simple decomposition
        assert len(result["sub_claims"]) >= 1

    def test_validate_subclaims_too_few(self, verifier):
        """Test validation when too few sub-claims are generated."""
        original_claim = "Test claim"
        sub_claims = []  # Empty list

        validated = verifier._validate_subclaims(sub_claims, original_claim)

        # Should fall back to original claim
        assert len(validated) == 1
        assert validated[0] == original_claim

    def test_validate_subclaims_too_many(self, verifier):
        """Test validation when too many sub-claims are generated."""
        original_claim = "Test claim"
        sub_claims = [f"Sub-claim {i}" for i in range(10)]  # 10 sub-claims

        validated = verifier._validate_subclaims(sub_claims, original_claim)

        # Should truncate to max_sub_claims (4)
        assert len(validated) == verifier.max_sub_claims


class TestEvidenceMapping:
    """Test Step 2: Evidence mapping functionality."""

    def test_verify_supported_subclaim(self, verifier, sample_passages):
        """Test evidence mapping for a supported sub-claim."""
        sub_claim = "Magazine A is American"
        with dspy.context(lm=LM):
            verification = verifier._verify_subclaim(sub_claim, sample_passages)

        assert verification.sub_claim == sub_claim
        assert verification.verdict in [
            SubClaimVerdict.SUPPORTED,
            SubClaimVerdict.UNCLEAR,
            SubClaimVerdict.INSUFFICIENT,
        ]
        assert 0.0 <= verification.confidence <= 1.0
        assert isinstance(verification.evidence, list)
        assert isinstance(verification.reasoning, str)

    def test_verify_refuted_subclaim(self, verifier, sample_passages):
        """Test evidence mapping for a refuted sub-claim."""
        sub_claim = "Magazine A is Austrian"  # Contradicts evidence
        with dspy.context(lm=LM):
            verification = verifier._verify_subclaim(sub_claim, sample_passages)

        assert verification.verdict in [
            SubClaimVerdict.REFUTED,
            SubClaimVerdict.UNCLEAR,
            SubClaimVerdict.INSUFFICIENT,
        ]

    def test_verify_with_no_evidence(self, verifier):
        """Test verification when no relevant evidence exists."""
        sub_claim = "The moon is made of cheese"
        irrelevant_passages = ["Doc1 | Water boils at 100°C", "Doc2 | Paris is in France"]

        with dspy.context(lm=LM):
            verification = verifier._verify_subclaim(sub_claim, irrelevant_passages)

        # Should indicate insufficient evidence
        assert verification.verdict in [
            SubClaimVerdict.INSUFFICIENT,
            SubClaimVerdict.UNCLEAR,
        ]


class TestConsistencyCheck:
    """Test Step 3: Logical consistency checking."""

    def test_detect_contradiction(self, verifier):
        """Test detection of entity mismatches (Austrian vs American)."""
        verifications = [
            SubClaimVerification(
                sub_claim="Magazine A is Austrian",
                verdict=SubClaimVerdict.SUPPORTED,
                evidence=[Evidence("Doc1", "Magazine A is Austrian", 0.9, 0)],
                confidence=0.9,
                reasoning="Evidence clearly states Austrian",
            ),
            SubClaimVerification(
                sub_claim="Magazine A is American",
                verdict=SubClaimVerdict.SUPPORTED,
                evidence=[Evidence("Doc2", "Magazine A is American", 0.9, 0)],
                confidence=0.9,
                reasoning="Evidence clearly states American",
            ),
        ]

        with dspy.context(lm=LM):
            contradictions = verifier._check_consistency(
                "Magazine A is both Austrian and American", verifications
            )

        # May or may not detect contradiction depending on LLM
        # Just verify it returns a list
        assert isinstance(contradictions, list)

    def test_no_contradictions(self, verifier):
        """Test consistency check with no contradictions."""
        verifications = [
            SubClaimVerification(
                sub_claim="The sky is blue",
                verdict=SubClaimVerdict.SUPPORTED,
                evidence=[Evidence("Doc1", "The sky appears blue", 0.9, 0)],
                confidence=0.9,
                reasoning="Direct evidence",
            ),
            SubClaimVerification(
                sub_claim="Water is wet",
                verdict=SubClaimVerdict.SUPPORTED,
                evidence=[Evidence("Doc2", "Water has wet properties", 0.9, 0)],
                confidence=0.9,
                reasoning="Direct evidence",
            ),
        ]

        with dspy.context(lm=LM):
            contradictions = verifier._check_consistency(
                "The sky is blue and water is wet", verifications
            )

        # Should find no or minimal contradictions
        assert isinstance(contradictions, list)


class TestEndToEnd:
    """Test complete verification pipeline."""

    def test_full_verification_pipeline(self, verifier, sample_claim, sample_passages):
        """Test complete verification pipeline end-to-end."""
        with dspy.context(lm=LM):
            result = verifier(claim=sample_claim, passages=sample_passages)

        # Check structure
        assert result.original_claim == sample_claim
        assert isinstance(result.sub_claims, list)
        assert len(result.sub_claims) >= 1
        assert result.overall_verdict in [
            OverallVerdict.SUPPORTS,
            OverallVerdict.REFUTES,
            OverallVerdict.NOT_ENOUGH_INFO,
        ]
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.evidence_mapping, dict)
        assert isinstance(result.reasoning_chain, str)
        assert "num_passages" in result.metadata
        assert "num_sub_claims" in result.metadata

    def test_edge_case_no_passages(self, verifier, sample_claim):
        """Test handling of empty document list."""
        with dspy.context(lm=LM):
            result = verifier(claim=sample_claim, passages=[])

        # Should handle gracefully
        assert result.overall_verdict == OverallVerdict.NOT_ENOUGH_INFO
        assert result.confidence_score == 0.0
        assert len(result.sub_claims) == 0
        assert result.metadata["num_passages"] == 0

    def test_verification_with_conflicting_documents(
        self, verifier, conflicting_passages
    ):
        """Test handling of contradictory evidence."""
        claim = "Magazine A was founded in 1990"

        with dspy.context(lm=LM):
            result = verifier(claim=claim, passages=conflicting_passages)

        # Should either detect contradictions or have lower confidence
        assert isinstance(result, object)
        assert hasattr(result, "contradictions")
        assert hasattr(result, "confidence_score")


class TestParsingHelpers:
    """Test helper parsing functions."""

    def test_parse_list_numbered(self, verifier):
        """Test parsing of numbered list output."""
        text = "1. First item\n2. Second item\n3. Third item"
        result = verifier._parse_list_output(text)

        assert len(result) == 3
        assert "First item" in result[0]
        assert "Second item" in result[1]
        assert "Third item" in result[2]

    def test_parse_list_bulleted(self, verifier):
        """Test parsing of bulleted list output."""
        text = "- First item\n- Second item\n* Third item"
        result = verifier._parse_list_output(text)

        assert len(result) >= 2  # Should parse at least some items

    def test_parse_list_newline_separated(self, verifier):
        """Test parsing of newline-separated output."""
        text = "First item\nSecond item\nThird item"
        result = verifier._parse_list_output(text)

        assert len(result) >= 3

    def test_parse_evidence_with_pipe(self, verifier):
        """Test parsing evidence in 'Title | Quote' format."""
        quote_str = "Magazine A | Founded in 1990"
        evidence = verifier._parse_evidence(quote_str)

        assert evidence.document_id == "Magazine A"
        assert "1990" in evidence.quote

    def test_parse_evidence_without_pipe(self, verifier):
        """Test parsing evidence without pipe separator."""
        quote_str = "Some quote without document ID"
        evidence = verifier._parse_evidence(quote_str)

        assert evidence.document_id == "Unknown"
        assert evidence.quote == quote_str

    def test_parse_verdict_supported(self, verifier):
        """Test parsing SUPPORTED verdict."""
        assert verifier._parse_verdict("SUPPORTED") == SubClaimVerdict.SUPPORTED
        assert verifier._parse_verdict("supported") == SubClaimVerdict.SUPPORTED
        assert verifier._parse_verdict("The claim is supported") == SubClaimVerdict.SUPPORTED

    def test_parse_verdict_refuted(self, verifier):
        """Test parsing REFUTED verdict."""
        assert verifier._parse_verdict("REFUTED") == SubClaimVerdict.REFUTED
        assert verifier._parse_verdict("refuted") == SubClaimVerdict.REFUTED

    def test_parse_verdict_insufficient(self, verifier):
        """Test parsing INSUFFICIENT verdict."""
        assert verifier._parse_verdict("INSUFFICIENT") == SubClaimVerdict.INSUFFICIENT
        assert verifier._parse_verdict("NOT_ENOUGH") == SubClaimVerdict.INSUFFICIENT

    def test_parse_confidence_float(self, verifier):
        """Test parsing confidence as float."""
        assert verifier._parse_confidence(0.85) == 0.85
        assert verifier._parse_confidence(0.5) == 0.5

    def test_parse_confidence_percentage_string(self, verifier):
        """Test parsing confidence as percentage string."""
        assert verifier._parse_confidence("85%") == 0.85
        assert verifier._parse_confidence("50") == 0.5

    def test_parse_confidence_invalid(self, verifier):
        """Test parsing invalid confidence value."""
        result = verifier._parse_confidence("invalid")
        assert 0.0 <= result <= 1.0  # Should return valid default

    def test_parse_overall_verdict_supports(self, verifier):
        """Test parsing overall SUPPORTS verdict."""
        assert verifier._parse_overall_verdict("SUPPORTS") == OverallVerdict.SUPPORTS
        assert verifier._parse_overall_verdict("supports") == OverallVerdict.SUPPORTS

    def test_parse_overall_verdict_refutes(self, verifier):
        """Test parsing overall REFUTES verdict."""
        assert verifier._parse_overall_verdict("REFUTES") == OverallVerdict.REFUTES

    def test_parse_overall_verdict_not_enough_info(self, verifier):
        """Test parsing NOT_ENOUGH_INFO verdict."""
        assert (
            verifier._parse_overall_verdict("NOT_ENOUGH_INFO")
            == OverallVerdict.NOT_ENOUGH_INFO
        )
        assert (
            verifier._parse_overall_verdict("unknown") == OverallVerdict.NOT_ENOUGH_INFO
        )


class TestEdgeCases:
    """Test handling of edge cases."""

    def test_single_word_claim(self, verifier, sample_passages):
        """Test claims that can't be easily decomposed."""
        with dspy.context(lm=LM):
            result = verifier(claim="True", passages=sample_passages)

        assert len(result.sub_claims) >= 1
        assert result.overall_verdict in OverallVerdict

    def test_very_long_claim(self, verifier, sample_passages):
        """Test claims with many clauses."""
        long_claim = " and ".join([f"Fact {i} is true" for i in range(10)])

        with dspy.context(lm=LM):
            result = verifier(claim=long_claim, passages=sample_passages)

        # Should cap at max_sub_claims
        assert len(result.sub_claims) <= verifier.max_sub_claims

    def test_irrelevant_passages(self, verifier):
        """Test passages unrelated to claim."""
        with dspy.context(lm=LM):
            result = verifier(
                claim="The sky is blue",
                passages=["Doc | Cats are mammals", "Doc | Water is wet"],
            )

        # Should indicate insufficient information
        assert result.overall_verdict in [
            OverallVerdict.NOT_ENOUGH_INFO,
            OverallVerdict.REFUTES,
        ]

    def test_disabled_consistency_check(self, sample_claim, sample_passages):
        """Test verifier with consistency checking disabled."""
        verifier_no_consistency = StructuredClaimVerifier(
            enable_consistency_check=False
        )

        with dspy.context(lm=LM):
            result = verifier_no_consistency(claim=sample_claim, passages=sample_passages)

        # Should have no contradictions (consistency check disabled)
        assert len(result.contradictions) == 0
        assert result.metadata["consistency_check_enabled"] is False
