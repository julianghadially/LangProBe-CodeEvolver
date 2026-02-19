import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverMultiHopWithVerification(LangProBeDSPyMetaProgram, dspy.Module):
    """Complete HoVer pipeline with structured verification.

    This module combines multi-hop retrieval with structured claim verification,
    enabling fine-grained verification and contradiction detection.

    Pipeline:
    1. HoverMultiHop: 3-hop retrieval (up to 21 documents)
    2. StructuredClaimVerifier: Decompose, verify, check consistency
    3. Return both retrieved docs and verification result

    Usage:
        program = HoverMultiHopWithVerification()
        result = program(claim="...")
        # Access: result.verification, result.retrieved_docs, result.label

    Args:
        enable_verification: Whether to enable verification (default: True)
        verifier_config: Optional configuration dict for StructuredClaimVerifier
    """

    def __init__(
        self, enable_verification: bool = True, verifier_config: dict = None
    ):
        super().__init__()
        self.retrieval = HoverMultiHop()
        self.enable_verification = enable_verification

        if enable_verification:
            from .hover_verifier import StructuredClaimVerifier

            verifier_config = verifier_config or {}
            self.verifier = StructuredClaimVerifier(**verifier_config)

    def forward(self, claim):
        """Forward pass combining retrieval and verification.

        Args:
            claim: The claim to verify

        Returns:
            dspy.Prediction with:
            - retrieved_docs: List of retrieved documents
            - verification: VerificationResult (if enabled)
            - label: Overall verdict value (if enabled)
        """
        # Step 1: Retrieve documents
        retrieval_result = self.retrieval(claim=claim)
        retrieved_docs = retrieval_result.retrieved_docs

        # Step 2: Verify (if enabled)
        if self.enable_verification:
            verification_result = self.verifier(claim=claim, passages=retrieved_docs)

            return dspy.Prediction(
                retrieved_docs=retrieved_docs,
                verification=verification_result,
                label=verification_result.overall_verdict.value,
            )
        else:
            return retrieval_result
