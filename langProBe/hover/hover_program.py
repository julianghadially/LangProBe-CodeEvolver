import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 10
        self.generate_queries_cot = dspy.ChainOfThought("claim -> reasoning, query1, query2, query3")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_docs_cot = dspy.ChainOfThought("claim, documents -> reasoning, top_documents")

    def forward(self, claim):
        # Generate 3 diverse queries using Chain-of-Thought reasoning
        # The reasoning explains what entities/facts need to be found
        cot_queries = self.generate_queries_cot(claim=claim)
        query1 = cot_queries.query1
        query2 = cot_queries.query2
        query3 = cot_queries.query3

        # Retrieve k=10 documents for each query (30 total documents)
        docs_query1 = self.retrieve_k(query1).passages
        docs_query2 = self.retrieve_k(query2).passages
        docs_query3 = self.retrieve_k(query3).passages

        # Combine all 30 documents
        all_docs = docs_query1 + docs_query2 + docs_query3

        # Rerank documents using Chain-of-Thought to select top 21 most relevant
        # The reasoning explains which documents are most relevant for verifying the claim
        documents_str = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(all_docs)])
        reranked = self.rerank_docs_cot(claim=claim, documents=documents_str)

        # Parse the top_documents from the reranking output
        # The reranked output should contain the selected documents
        top_docs = reranked.top_documents

        return dspy.Prediction(retrieved_docs=all_docs, reranked_docs=top_docs, reasoning=reranked.reasoning)


