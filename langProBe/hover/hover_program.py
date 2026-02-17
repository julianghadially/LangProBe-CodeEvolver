import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class RerankDocuments(dspy.Signature):
    """Score each document's relevance to the claim. Return a relevance score for each document."""
    claim: str = dspy.InputField()
    documents: list[str] = dspy.InputField()
    relevance_scores: list[float] = dspy.OutputField()

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.reranker = dspy.ChainOfThought(RerankDocuments)

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

        # Collect all 45 candidate documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate documents by content
        seen = set()
        deduplicated_docs = []
        for doc in all_docs:
            if doc not in seen:
                seen.add(doc)
                deduplicated_docs.append(doc)

        # Rerank documents using the reranker
        rerank_result = self.reranker(claim=claim, documents=deduplicated_docs)
        relevance_scores = rerank_result.relevance_scores

        # Sort documents by relevance scores descending
        doc_score_pairs = list(zip(deduplicated_docs, relevance_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top 21 unique documents
        top_docs = [doc for doc, score in doc_score_pairs[:21]]

        return dspy.Prediction(retrieved_docs=top_docs)
