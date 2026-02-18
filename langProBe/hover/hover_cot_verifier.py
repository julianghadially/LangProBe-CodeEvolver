"""Chain-of-Thought Verifier Module for HoVer Benchmark.

This module implements a multi-step reasoning pipeline that sits between document
retrieval and final answer generation. It explicitly extracts facts, performs
step-by-step logical reasoning, and outputs a verification decision with
supporting reasoning.
"""

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class FactExtractionSignature(dspy.Signature):
    """Extract key facts from retrieved documents into structured statements.

    Parse retrieved documents and extract atomic facts that are relevant to the claim.
    Each fact should be a clear, standalone statement that can be used in reasoning.
    """

    claim = dspy.InputField(desc="The claim to verify")
    documents = dspy.InputField(desc="Retrieved documents with relevant information")
    facts: list[str] = dspy.OutputField(
        desc="List of key facts extracted from documents, each as a clear statement. "
        "Each fact should be atomic and directly related to verifying the claim."
    )


class MultiHopReasoningSignature(dspy.Signature):
    """Perform step-by-step logical reasoning to connect multi-hop facts.

    Connect facts through logical inference chains. For example:
    - Fact 1: "Erik Watts won TCW Tag Team Championship"
    - Fact 2: "Bill Watts is Erik Watts' father"
    - Reasoning: "Therefore, Bill Watts is father of TCW winner"
    """

    claim = dspy.InputField(desc="The claim to verify")
    facts = dspy.InputField(desc="Extracted facts from documents")
    reasoning_steps: list[str] = dspy.OutputField(
        desc="Step-by-step reasoning process showing how facts connect. "
        "Each step should show explicit logical connections, comparisons, or calculations. "
        "Build up the reasoning chain systematically."
    )


class ExplicitComparisonSignature(dspy.Signature):
    """Perform explicit comparisons or calculations when needed.

    Handle explicit verification requirements such as:
    - Counting: "Band has 3 members: John, Paul, George → Claim says 4 → NOT_SUPPORTED"
    - Dates: "Event happened in 1995 → Claim says 1990 → NOT_SUPPORTED"
    - Locations: "City is in California → Claim says Texas → NOT_SUPPORTED"
    """

    claim = dspy.InputField(desc="The claim to verify")
    reasoning_steps = dspy.InputField(desc="Reasoning steps performed so far")
    facts = dspy.InputField(desc="Extracted facts from documents")
    comparisons: list[str] = dspy.OutputField(
        desc="Explicit comparisons, calculations, or verifications. "
        "Examples: counting members, comparing dates/numbers, verifying locations. "
        "Show the explicit comparison and its result."
    )


class FinalVerificationSignature(dspy.Signature):
    """Make final verification decision based on all reasoning.

    Synthesize all reasoning into a final binary decision with explanation.
    Consider all facts, reasoning steps, and comparisons to determine if the
    claim is fully supported by the evidence.
    """

    claim = dspy.InputField(desc="The claim to verify")
    facts = dspy.InputField(desc="Extracted facts from documents")
    reasoning_steps = dspy.InputField(desc="Step-by-step reasoning performed")
    comparisons = dspy.InputField(desc="Explicit comparisons and calculations")
    verification_decision = dspy.OutputField(
        desc="Final decision: SUPPORTED if claim is fully supported by facts, "
        "NOT_SUPPORTED otherwise. Must include brief justification explaining "
        "why the claim is or is not supported based on the evidence."
    )
    label: int = dspy.OutputField(
        desc="Binary label: 1 if claim is SUPPORTED, 0 if NOT_SUPPORTED"
    )


class ChainOfThoughtVerifier(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-step reasoning verifier using Chain-of-Thought prompting.

    This module takes a claim and retrieved documents, then performs four stages
    of reasoning:

    1. Fact Extraction: Parse documents into atomic facts
    2. Multi-Hop Reasoning: Connect facts through logical inference
    3. Explicit Comparisons: Handle numerical/spatial/temporal verification
    4. Final Verification: Synthesize into binary label with explanation

    Each stage uses Chain-of-Thought prompting to force the LLM to show its work
    before outputting results, improving transparency and accuracy.
    """

    def __init__(self):
        super().__init__()

        # Step 1: Fact Extraction
        self.extract_facts = dspy.ChainOfThought(FactExtractionSignature)

        # Step 2: Multi-Hop Reasoning
        self.perform_reasoning = dspy.ChainOfThought(MultiHopReasoningSignature)

        # Step 3: Explicit Comparisons
        self.perform_comparisons = dspy.ChainOfThought(ExplicitComparisonSignature)

        # Step 4: Final Verification
        self.final_verification = dspy.ChainOfThought(FinalVerificationSignature)

    def forward(self, claim, retrieved_docs):
        """Execute the four-stage Chain-of-Thought verification pipeline.

        Args:
            claim: The claim to verify
            retrieved_docs: List of retrieved documents (strings or passages)

        Returns:
            dspy.Prediction with:
                - label: Binary label (0 or 1)
                - verification_decision: Explanation of the decision
                - facts: Extracted facts from documents
                - reasoning_steps: Step-by-step reasoning
                - comparisons: Explicit comparisons/calculations
        """
        # STEP 1: Extract facts from documents
        facts_pred = self.extract_facts(claim=claim, documents=retrieved_docs)
        facts = facts_pred.facts

        # STEP 2: Perform multi-hop reasoning
        reasoning_pred = self.perform_reasoning(claim=claim, facts=facts)
        reasoning_steps = reasoning_pred.reasoning_steps

        # STEP 3: Perform explicit comparisons/calculations
        comparisons_pred = self.perform_comparisons(
            claim=claim, reasoning_steps=reasoning_steps, facts=facts
        )
        comparisons = comparisons_pred.comparisons

        # STEP 4: Final verification decision
        verification_pred = self.final_verification(
            claim=claim,
            facts=facts,
            reasoning_steps=reasoning_steps,
            comparisons=comparisons,
        )

        return dspy.Prediction(
            label=verification_pred.label,
            verification_decision=verification_pred.verification_decision,
            facts=facts,
            reasoning_steps=reasoning_steps,
            comparisons=comparisons,
        )
