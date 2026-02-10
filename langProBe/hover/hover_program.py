import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 10  # Retrieve 10 docs per hop for 30 total candidates
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")
        self.reranker = dspy.ChainOfThought("claim, document -> relevance_score: float")

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

        # Stage 2: Rerank all 30 candidate documents
        all_candidate_docs = hop1_docs + hop2_docs + hop3_docs
        doc_scores = []

        for doc in all_candidate_docs:
            result = self.reranker(claim=claim, document=doc)
            score = float(result.relevance_score)
            doc_scores.append((doc, score))

        # Sort by relevance score (descending) and select top 21
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_21_docs = [doc for doc, score in doc_scores[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)


