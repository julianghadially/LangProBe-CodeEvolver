import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from typing import Literal


class ExtractEntitiesSignature(dspy.Signature):
    """
    Extract the key entities, concepts, and relationships that must be verified to
    evaluate the claim. Focus on named entities (people, organizations, locations,
    events) and specific facts that require evidence. Limit to the 10 most important
    entities, prioritizing named entities and specific facts over general concepts.
    """

    claim = dspy.InputField(desc="The claim to analyze")
    entities: list[str] = dspy.OutputField(
        desc="List of key entities, concepts, and facts that need evidence. "
        "Each entity should be specific and searchable (e.g., 'Barack Obama', "
        "'2008 presidential election', 'Nobel Peace Prize 2009')"
    )


class AnalyzeCoverageSignature(dspy.Signature):
    """
    Analyze which entities and concepts are covered by the retrieved passages
    and which are still missing. An entity is 'covered' if there is substantive
    information about it in the passages, not just a passing mention.
    """

    claim = dspy.InputField(desc="The original claim")
    required_entities = dspy.InputField(
        desc="List of entities that need to be covered"
    )
    passages = dspy.InputField(
        desc="Retrieved passages from previous hop(s)"
    )
    covered_entities: list[str] = dspy.OutputField(
        desc="Entities that have substantive coverage in the passages"
    )
    missing_entities: list[str] = dspy.OutputField(
        desc="Entities that are not covered or only mentioned in passing"
    )
    coverage_summary = dspy.OutputField(
        desc="Brief summary of what information is present and what gaps remain"
    )


class GenerateGapQuerySignature(dspy.Signature):
    """
    Generate a search query specifically targeting the missing entities and information
    gaps. The query should be focused and likely to retrieve documents that address
    the uncovered aspects of the claim.
    """

    claim = dspy.InputField(desc="The original claim")
    missing_entities = dspy.InputField(
        desc="Entities and concepts that are not yet covered"
    )
    coverage_summary = dspy.InputField(
        desc="Summary of what information is present and what gaps remain"
    )
    context = dspy.InputField(
        desc="Information already gathered from previous hops",
        default=""
    )
    query = dspy.OutputField(
        desc="Search query targeting the missing information. Should be specific "
        "and focused on retrieving documents about the uncovered entities."
    )


class GapAnalysisModule(dspy.Module):
    """Module for gap-aware entity tracking and targeted query generation."""

    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(ExtractEntitiesSignature)
        self.analyze_coverage = dspy.ChainOfThought(AnalyzeCoverageSignature)
        self.generate_gap_query = dspy.ChainOfThought(GenerateGapQuerySignature)


class ExtractKeyFactsSignature(dspy.Signature):
    """
    Extract key facts from the retrieved documents that are relevant to verifying
    the claim. Focus on specific factual statements that can support or refute
    elements of the claim. Each fact should be self-contained and verifiable.
    """

    claim = dspy.InputField(desc="The claim to verify")
    passages = dspy.InputField(desc="Retrieved documents/passages to analyze")
    key_facts: list[str] = dspy.OutputField(
        desc="List of key facts extracted from the passages that are relevant to the claim. "
        "Each fact should be specific, factual, and directly relevant to verification."
    )


class IdentifyMissingInfoSignature(dspy.Signature):
    """
    Perform gap analysis to identify what information is still missing to fully
    verify or refute the claim. Compare the claim's assertions with the extracted
    facts to determine what gaps exist in our evidence.
    """

    claim = dspy.InputField(desc="The claim to verify")
    key_facts = dspy.InputField(desc="Key facts extracted from retrieved documents")
    missing_info: list[str] = dspy.OutputField(
        desc="List of specific pieces of information that are missing or not adequately "
        "covered by the extracted facts. Focus on elements mentioned in the claim that "
        "lack supporting or refuting evidence."
    )
    coverage_assessment = dspy.OutputField(
        desc="Brief assessment of how well the extracted facts cover the claim's assertions"
    )


class ChainFactsSignature(dspy.Signature):
    """
    Chain together the extracted facts to form logical connections that relate to
    the claim. Identify how different facts connect to each other and to specific
    parts of the claim. Build reasoning chains that lead toward a verification decision.
    """

    claim = dspy.InputField(desc="The claim to verify")
    key_facts = dspy.InputField(desc="Key facts extracted from documents")
    missing_info = dspy.InputField(desc="Information identified as missing")
    reasoning_chains: list[str] = dspy.OutputField(
        desc="List of logical reasoning chains connecting the facts. Each chain should "
        "explain how facts connect to each other and to the claim, building toward "
        "a verification decision. Include both supporting and refuting connections."
    )


class FinalVerificationSignature(dspy.Signature):
    """
    Make a final verification decision based on the extracted facts, gap analysis,
    and reasoning chains. Determine whether the claim is SUPPORTED or REFUTED by
    the evidence, and provide a confidence score reflecting the strength of the evidence
    and the impact of any missing information.
    """

    claim = dspy.InputField(desc="The claim to verify")
    key_facts = dspy.InputField(desc="Key facts extracted from documents")
    missing_info = dspy.InputField(desc="Information identified as missing")
    reasoning_chains = dspy.InputField(desc="Logical reasoning chains connecting facts")
    decision: Literal["SUPPORTS", "REFUTES"] = dspy.OutputField(
        desc="Final verification decision: SUPPORTS if the evidence confirms the claim, "
        "REFUTES if the evidence contradicts the claim"
    )
    confidence_score: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0 reflecting the strength of the evidence. "
        "Consider factors like: completeness of evidence, consistency of facts, "
        "impact of missing information, and strength of reasoning chains."
    )
    justification = dspy.OutputField(
        desc="Clear explanation of the verification decision, referencing specific facts "
        "and reasoning chains that led to the conclusion"
    )


class ChainOfThoughtVerifier(dspy.Module):
    """
    Chain-of-thought reasoning module that processes retrieved documents to verify claims.

    The verifier performs a structured reasoning process:
    1. Extracts key facts from relevant documents
    2. Identifies missing information (gap analysis)
    3. Chains facts together to form logical connections
    4. Outputs final verification decision (SUPPORTS/REFUTES) with confidence score
    """

    def __init__(self):
        super().__init__()
        self.extract_facts = dspy.ChainOfThought(ExtractKeyFactsSignature)
        self.identify_gaps = dspy.ChainOfThought(IdentifyMissingInfoSignature)
        self.chain_facts = dspy.ChainOfThought(ChainFactsSignature)
        self.final_verification = dspy.ChainOfThought(FinalVerificationSignature)

    def forward(self, claim, passages):
        """
        Verify a claim using chain-of-thought reasoning over retrieved passages.

        Args:
            claim: The claim to verify
            passages: List of retrieved document passages

        Returns:
            dspy.Prediction with decision, confidence_score, justification, and intermediate reasoning
        """
        # Step 1: Extract key facts from passages
        facts_result = self.extract_facts(claim=claim, passages=passages)
        key_facts = facts_result.key_facts

        # Step 2: Identify missing information (gap analysis)
        gaps_result = self.identify_gaps(claim=claim, key_facts=key_facts)
        missing_info = gaps_result.missing_info
        coverage_assessment = gaps_result.coverage_assessment

        # Step 3: Chain facts together to form logical connections
        chains_result = self.chain_facts(
            claim=claim,
            key_facts=key_facts,
            missing_info=missing_info
        )
        reasoning_chains = chains_result.reasoning_chains

        # Step 4: Make final verification decision
        verification_result = self.final_verification(
            claim=claim,
            key_facts=key_facts,
            missing_info=missing_info,
            reasoning_chains=reasoning_chains
        )

        # Return comprehensive prediction with all reasoning steps
        return dspy.Prediction(
            decision=verification_result.decision,
            confidence_score=verification_result.confidence_score,
            justification=verification_result.justification,
            key_facts=key_facts,
            missing_info=missing_info,
            coverage_assessment=coverage_assessment,
            reasoning_chains=reasoning_chains
        )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7

        # NEW: Gap analysis module
        self.gap_analyzer = GapAnalysisModule()

        # EXISTING: Keep for backward compatibility and fallback
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # STEP 1: Extract required entities from claim (once, at the start)
        entity_extraction = self.gap_analyzer.extract_entities(claim=claim)
        required_entities = entity_extraction.entities

        # Handle edge case: no entities extracted
        if not required_entities:
            required_entities = [claim]  # Treat entire claim as single entity

        # HOP 1: Initial retrieval based on claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

        # STEP 2: Analyze coverage after hop 1
        coverage_hop1 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=hop1_docs
        )

        # HOP 2: Generate gap-targeted query or fallback to original
        if coverage_hop1.missing_entities:
            hop2_query = self.gap_analyzer.generate_gap_query(
                claim=claim,
                missing_entities=coverage_hop1.missing_entities,
                coverage_summary=coverage_hop1.coverage_summary,
                context=summary_1
            ).query
        else:
            # Fallback: use existing query generator if all entities covered
            hop2_query = self.create_query_hop2(
                claim=claim, summary_1=summary_1
            ).query

        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # STEP 3: Analyze coverage after hop 2 (on all passages so far)
        all_passages_so_far = hop1_docs + hop2_docs
        coverage_hop2 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=all_passages_so_far
        )

        # HOP 3: Generate gap-targeted query for remaining gaps or fallback
        if coverage_hop2.missing_entities:
            hop3_query = self.gap_analyzer.generate_gap_query(
                claim=claim,
                missing_entities=coverage_hop2.missing_entities,
                coverage_summary=coverage_hop2.coverage_summary,
                context=summary_1 + " " + summary_2
            ).query
        else:
            # Fallback: broaden search if all entities covered
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query

        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverProgram(LangProBeDSPyMetaProgram, dspy.Module):
    """
    Complete Hover program that retrieves documents AND verifies claims using
    chain-of-thought reasoning.

    EVALUATION
    - This system retrieves relevant documents and then verifies the claim
    - Returns verification decision (SUPPORTS/REFUTES) with confidence score
    - Uses explicit reasoning that connects supporting facts before classification
    """

    def __init__(self):
        super().__init__()
        # Reuse the existing multi-hop retrieval module
        self.retriever = HoverMultiHop()
        # Add chain-of-thought verifier
        self.verifier = ChainOfThoughtVerifier()

    def forward(self, claim):
        """
        Retrieve documents and verify the claim with chain-of-thought reasoning.

        Args:
            claim: The claim to verify

        Returns:
            dspy.Prediction with:
                - retrieved_docs: Documents retrieved (for backward compatibility)
                - decision: SUPPORTS or REFUTES
                - confidence_score: Float between 0.0 and 1.0
                - justification: Explanation of the decision
                - key_facts: Facts extracted from documents
                - missing_info: Gap analysis results
                - reasoning_chains: Logical connections between facts
        """
        # Step 1: Retrieve relevant documents using multi-hop retrieval
        retrieval_result = self.retriever(claim=claim)
        retrieved_docs = retrieval_result.retrieved_docs

        # Step 2: Verify the claim using chain-of-thought reasoning
        verification_result = self.verifier(
            claim=claim,
            passages=retrieved_docs
        )

        # Return combined prediction with both retrieval and verification results
        return dspy.Prediction(
            retrieved_docs=retrieved_docs,
            decision=verification_result.decision,
            confidence_score=verification_result.confidence_score,
            justification=verification_result.justification,
            key_facts=verification_result.key_facts,
            missing_info=verification_result.missing_info,
            coverage_assessment=verification_result.coverage_assessment,
            reasoning_chains=verification_result.reasoning_chains
        )
