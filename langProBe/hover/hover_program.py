import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimDecomposer(dspy.Signature):
    """Decompose a claim into 2-3 atomic sub-hypotheses that need independent verification.
    Each hypothesis should represent a specific fact or entity relationship that can be verified with documents."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    hypotheses: list[str] = dspy.OutputField(desc="2-3 atomic sub-hypotheses that need verification (e.g., 'Hypothesis 1: Paul Éluard wrote Capitale de la douleur', 'Hypothesis 2: Paul Éluard was French')")


class HypothesisVerifier(dspy.Signature):
    """Verify which hypotheses are supported by the retrieved documents and identify missing information.
    Assess the evidence quality and identify any entities or facts that still need retrieval."""

    claim: str = dspy.InputField(desc="The original claim")
    hypotheses: list[str] = dspy.InputField(desc="The sub-hypotheses to verify")
    retrieved_docs: str = dspy.InputField(desc="The documents retrieved so far")
    verification_status: str = dspy.OutputField(desc="Explanation of which hypotheses are supported/unsupported by current documents")
    missing_entities: list[str] = dspy.OutputField(desc="List of specific entities or facts that need additional retrieval (empty list if all verified)")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1 that all hypotheses are sufficiently verified")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using hypothesis-driven retrieval with self-verification.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Stage 1: Initial retrieval
        self.decomposer = dspy.ChainOfThought(ClaimDecomposer)
        self.retrieve_initial = dspy.Retrieve(k=15)

        # Stage 2: Verification and targeted retrieval
        self.verifier = dspy.ChainOfThought(HypothesisVerifier)
        self.retrieve_targeted = dspy.Retrieve(k=3)

    def forward(self, claim):
        # STAGE 1: Hypothesis Decomposition & Initial Retrieval
        # Decompose claim into atomic sub-hypotheses
        decomposition = self.decomposer(claim=claim)
        hypotheses = decomposition.hypotheses

        # Generate comprehensive query covering all hypotheses
        # Combine claim with hypotheses for a rich initial query
        comprehensive_query = f"{claim} {' '.join(hypotheses)}"

        # Initial retrieval with k=15
        initial_docs = self.retrieve_initial(comprehensive_query).passages

        # STAGE 2: Self-Verification & Targeted Retrieval
        # Format documents for verification
        docs_text = "\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(initial_docs)])

        # Verify hypotheses coverage
        verification = self.verifier(
            claim=claim,
            hypotheses=hypotheses,
            retrieved_docs=docs_text
        )

        all_docs = initial_docs.copy()

        # If confidence < 0.8 or missing entities exist, perform targeted retrieval
        confidence = verification.confidence
        missing_entities = verification.missing_entities

        # Perform up to 2 targeted retrievals for missing entities
        if confidence < 0.8 or (missing_entities and len(missing_entities) > 0):
            # First targeted retrieval
            if len(missing_entities) > 0:
                target_query_1 = " ".join(missing_entities[:2])  # Use first 1-2 missing entities
                targeted_docs_1 = self.retrieve_targeted(target_query_1).passages
                all_docs.extend(targeted_docs_1)

                # Second targeted retrieval if more entities remain
                if len(missing_entities) > 2:
                    target_query_2 = " ".join(missing_entities[2:4])  # Use next 1-2 missing entities
                    targeted_docs_2 = self.retrieve_targeted(target_query_2).passages
                    all_docs.extend(targeted_docs_2)

        return dspy.Prediction(retrieved_docs=all_docs)
