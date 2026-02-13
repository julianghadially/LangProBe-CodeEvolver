import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 10
        self.assess_relevance = dspy.Predict("claim, passages -> relevant_findings: str")
        self.create_query_hop2 = dspy.Predict("claim, relevant_findings_1 -> query")
        self.create_query_hop3 = dspy.Predict("claim, relevant_findings_1, relevant_findings_2 -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank = dspy.Predict("claim, all_passages -> ranked_passage_indices: list[int]")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        relevant_findings_1 = self.assess_relevance(
            claim=claim, passages=hop1_docs
        ).relevant_findings

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, relevant_findings_1=relevant_findings_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        relevant_findings_2 = self.assess_relevance(
            claim=claim, passages=hop2_docs
        ).relevant_findings

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, relevant_findings_1=relevant_findings_1, relevant_findings_2=relevant_findings_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all 30 documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Rerank to select top 21 most relevant documents
        rerank_result = self.rerank(claim=claim, all_passages=all_docs)
        ranked_indices = rerank_result.ranked_passage_indices

        # Select top 21 documents in order of relevance
        top_21_docs = [all_docs[i] for i in ranked_indices[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)


