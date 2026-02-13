import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.generate_parallel_queries = dspy.Predict("claim->query1,query2,query3")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Generate 3 diverse queries in parallel from the claim
        parallel_queries = self.generate_parallel_queries(claim=claim)
        query1 = parallel_queries.query1
        query2 = parallel_queries.query2
        query3 = parallel_queries.query3

        # Retrieve k=7 documents for each query (21 total documents)
        docs_query1 = self.retrieve_k(query1).passages
        docs_query2 = self.retrieve_k(query2).passages
        docs_query3 = self.retrieve_k(query3).passages

        # Combine all 21 documents
        all_docs = docs_query1 + docs_query2 + docs_query3

        return dspy.Prediction(retrieved_docs=all_docs)


