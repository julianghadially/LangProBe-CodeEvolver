import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.select_docs_hop1 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.select_docs_hop2 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.create_query_hop2 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        selection_1 = self.select_docs_hop1(
            claim=claim, passages=hop1_docs
        )
        selected_passages_1 = selection_1.selected_passages

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, selected_passages=selected_passages_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        selection_2 = self.select_docs_hop2(
            claim=claim, passages=hop2_docs
        )
        selected_passages_2 = selection_2.selected_passages

        # HOP 3
        # Combine selected passages from hop1 and hop2 for context
        combined_selected = selected_passages_1 + selected_passages_2
        hop3_query = self.create_query_hop3(
            claim=claim, selected_passages=combined_selected
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


