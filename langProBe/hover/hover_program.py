import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        # Chain-of-thought reasoning modules
        self.query_planner = dspy.ChainOfThought(
            "claim -> reasoning, key_entities, connection_path, query1, query2, query3"
        )
        self.next_query_reasoner = dspy.ChainOfThought(
            "claim, retrieved_context, previous_reasoning -> reasoning, missing_info, query"
        )
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # PLANNING PHASE: Reason about retrieval strategy
        plan = self.query_planner(claim=claim)

        # HOP 1: Use first planned query
        hop1_query = plan.query1
        hop1_docs = self.retrieve_k(hop1_query).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Reason about what's missing and generate next query
        hop2_reasoning = self.next_query_reasoner(
            claim=claim,
            retrieved_context=summary_1,
            previous_reasoning=plan.reasoning
        )
        hop2_query = hop2_reasoning.query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Reason about final missing information
        hop3_reasoning = self.next_query_reasoner(
            claim=claim,
            retrieved_context=f"{summary_1} {summary_2}",
            previous_reasoning=hop2_reasoning.reasoning
        )
        hop3_query = hop3_reasoning.query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


