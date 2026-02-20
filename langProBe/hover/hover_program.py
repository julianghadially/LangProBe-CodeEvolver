import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapAnalysis(dspy.Signature):
    """Analyze the gap between what the claim requires for verification and what has been retrieved so far.
    Identify key entities, topics, or facts that are still missing to fully verify or refute the claim."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    retrieved_passages: list[str] = dspy.InputField(desc="passages retrieved so far from previous hops")
    missing_entities: list[str] = dspy.OutputField(desc="list of key entities, topics, or facts not yet covered that are needed to verify the claim")
    coverage_assessment: str = dspy.OutputField(desc="assessment of what is already covered versus what is still missing")


class CreateQueryHop2(dspy.Signature):
    """Generate a targeted search query for the second retrieval hop, focusing on missing entities and gaps identified after the first hop."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    summary_1: str = dspy.InputField(desc="summary of documents from hop 1")
    missing_entities: list[str] = dspy.InputField(desc="key entities or topics missing from hop 1 that need to be found")
    query: str = dspy.OutputField(desc="targeted search query for hop 2 that addresses the gaps")


class CreateQueryHop3(dspy.Signature):
    """Generate a highly targeted search query for the third retrieval hop, focusing on remaining gaps after two hops."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    summary_1: str = dspy.InputField(desc="summary of documents from hop 1")
    summary_2: str = dspy.InputField(desc="summary of documents from hop 2")
    missing_entities: list[str] = dspy.InputField(desc="key entities or topics still missing after hop 2 that need to be found")
    query: str = dspy.OutputField(desc="highly targeted search query for hop 3 that addresses the remaining gaps")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        # Gap analysis modules
        self.gap_analysis_hop1 = dspy.ChainOfThought(GapAnalysis)
        self.gap_analysis_hop2 = dspy.ChainOfThought(GapAnalysis)
        # Query generation with missing entities
        self.create_query_hop2 = dspy.ChainOfThought(CreateQueryHop2)
        self.create_query_hop3 = dspy.ChainOfThought(CreateQueryHop3)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # GAP ANALYSIS AFTER HOP 1
        gap1_result = self.gap_analysis_hop1(
            claim=claim,
            retrieved_passages=hop1_docs
        )
        missing_entities_hop1 = gap1_result.missing_entities

        # HOP 2 - Use gap analysis to generate targeted query
        hop2_query = self.create_query_hop2(
            claim=claim,
            summary_1=summary_1,
            missing_entities=missing_entities_hop1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # GAP ANALYSIS AFTER HOP 2
        all_docs_so_far = hop1_docs + hop2_docs
        gap2_result = self.gap_analysis_hop2(
            claim=claim,
            retrieved_passages=all_docs_so_far
        )
        missing_entities_hop2 = gap2_result.missing_entities

        # HOP 3 - Use gap analysis to generate highly targeted query
        hop3_query = self.create_query_hop3(
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            missing_entities=missing_entities_hop2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
