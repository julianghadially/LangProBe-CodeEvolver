import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        # Gap analysis modules - use ChainOfThought for identifying missing information
        self.gap_analysis_hop1 = dspy.ChainOfThought("claim, retrieved_passages -> missing_info")
        self.gap_analysis_hop2 = dspy.ChainOfThought("claim, all_retrieved_passages -> missing_info")

        # Query generation modules that use gap analysis
        self.create_query_hop2 = dspy.Predict("claim, retrieved_passages, missing_info -> query")
        self.create_query_hop3 = dspy.Predict("claim, all_retrieved_passages, missing_info -> query")

        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1: Initial retrieval using the claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # GAP ANALYSIS AFTER HOP 1: Identify what critical information is still missing
        # This uses ChainOfThought to reason about what entities, facts, or connections
        # are needed to verify the claim based on what was already retrieved
        gap_after_hop1 = self.gap_analysis_hop1(
            claim=claim, retrieved_passages=hop1_docs
        ).missing_info

        # HOP 2: Generate query targeting the identified gaps, not just summarizing what was found
        hop2_query = self.create_query_hop2(
            claim=claim, retrieved_passages=hop1_docs, missing_info=gap_after_hop1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # GAP ANALYSIS AFTER HOP 2: Identify remaining gaps based on all retrieved passages
        all_docs_hop2 = hop1_docs + hop2_docs
        gap_after_hop2 = self.gap_analysis_hop2(
            claim=claim, all_retrieved_passages=all_docs_hop2
        ).missing_info

        # HOP 3: Final query targeting any remaining information gaps
        # This ensures we find connecting documents needed for multi-hop reasoning
        hop3_query = self.create_query_hop3(
            claim=claim, all_retrieved_passages=all_docs_hop2, missing_info=gap_after_hop2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


