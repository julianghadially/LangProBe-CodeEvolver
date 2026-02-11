import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 10
        self.select_docs_hop1 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.select_docs_hop2 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.create_query_hop2 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_document = dspy.ChainOfThought("claim, document -> relevance_score: float, reasoning: str")
        self.rerank_document.__doc__ = "Score the document's relevance to verifying the claim. Return a relevance_score between 0.0 and 1.0, where 1.0 means highly relevant for claim verification."

    def forward(self, claim):
        # HOP 1: Retrieve k=10 documents based on claim
        hop1_docs = self.retrieve_k(claim).passages
        selection_1 = self.select_docs_hop1(
            claim=claim, passages=hop1_docs
        )
        selected_passages_1 = selection_1.selected_passages

        # HOP 2: Create query based on hop1 results and retrieve k=10 documents
        hop2_query = self.create_query_hop2(
            claim=claim, selected_passages=selected_passages_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        selection_2 = self.select_docs_hop2(
            claim=claim, passages=hop2_docs
        )
        selected_passages_2 = selection_2.selected_passages

        # HOP 3: Create query based on hop1 and hop2 results and retrieve k=10 documents
        combined_selected = selected_passages_1 + selected_passages_2
        hop3_query = self.create_query_hop3(
            claim=claim, selected_passages=combined_selected
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all 30 documents from 3 hops and deduplicate
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = []
        seen = set()
        for doc in all_docs:
            if doc not in seen:
                unique_docs.append(doc)
                seen.add(doc)

        # RERANKING: Score each document for relevance to claim verification
        scored_docs = []
        for doc in unique_docs:
            rerank_result = self.rerank_document(claim=claim, document=doc)
            try:
                score = float(rerank_result.relevance_score)
            except (ValueError, TypeError):
                # Default to 0.5 if score parsing fails
                score = 0.5
            scored_docs.append((score, doc))

        # Sort by relevance score (highest first) and select top 21 documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_21_docs = [doc for score, doc in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)


