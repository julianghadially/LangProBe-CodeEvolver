import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 15
        self.generate_parallel_queries = dspy.Predict("claim->query1,query2,query3")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_documents = dspy.ChainOfThought("claim, documents -> top_21_doc_indices")

    def forward(self, claim):
        # Generate 3 diverse queries in parallel from the claim
        parallel_queries = self.generate_parallel_queries(claim=claim)
        query1 = parallel_queries.query1
        query2 = parallel_queries.query2
        query3 = parallel_queries.query3

        # Retrieve k=15 documents for each query (45 total documents)
        docs_query1 = self.retrieve_k(query1).passages
        docs_query2 = self.retrieve_k(query2).passages
        docs_query3 = self.retrieve_k(query3).passages

        # Combine all 45 documents
        all_docs = docs_query1 + docs_query2 + docs_query3

        # Rerank all 45 documents to select the top 21 most relevant for multi-hop reasoning
        # Format documents for reranking with indices
        docs_with_indices = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(all_docs)])
        rerank_result = self.rerank_documents(claim=claim, documents=docs_with_indices)

        # Parse the top 21 document indices from the reranker output
        top_indices = self._parse_indices(rerank_result.top_21_doc_indices, max_docs=len(all_docs))

        # Select the top 21 documents based on reranker's indices
        reranked_docs = [all_docs[i] for i in top_indices if i < len(all_docs)][:21]

        return dspy.Prediction(retrieved_docs=reranked_docs)

    def _parse_indices(self, indices_str, max_docs):
        """Parse comma-separated indices from reranker output, handling various formats."""
        indices = []
        # Try to extract numbers from the string
        import re
        numbers = re.findall(r'\d+', str(indices_str))
        for num_str in numbers:
            idx = int(num_str)
            if 0 <= idx < max_docs and idx not in indices:
                indices.append(idx)
            if len(indices) >= 21:
                break
        # If we couldn't parse enough indices, fall back to first 21 documents
        if len(indices) < 21:
            indices = list(range(min(21, max_docs)))
        return indices


