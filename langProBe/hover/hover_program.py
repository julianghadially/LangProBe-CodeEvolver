import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimQueryDecomposerSignature(dspy.Signature):
    """Generate 3 diverse search queries that focus on different entities and aspects mentioned in the claim.
    Each query should target a specific entity, concept, or relationship in the claim to ensure comprehensive coverage.
    For example, for a claim like 'The director of Film X was born in Country Y', generate:
    - query1: Focus on the film and its director
    - query2: Focus on the film itself
    - query3: Focus on birthplace information"""

    claim = dspy.InputField(desc="The claim to decompose into search queries")
    query1 = dspy.OutputField(desc="First search query focusing on a specific entity or aspect")
    query2 = dspy.OutputField(desc="Second search query focusing on a different entity or aspect")
    query3 = dspy.OutputField(desc="Third search query focusing on yet another entity or aspect")


class ClaimQueryDecomposer(dspy.Module):
    """DSPy module that decomposes a claim into 3 diverse search queries."""

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(ClaimQueryDecomposerSignature)

    def forward(self, claim):
        result = self.decompose(claim=claim)
        return dspy.Prediction(
            query1=result.query1,
            query2=result.query2,
            query3=result.query3
        )




class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using query decomposition and parallel retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Uses query decomposition to generate 3 diverse queries targeting different aspects of the claim
    - Retrieves 25 documents per query in parallel (total 75 documents)
    - Applies Reciprocal Rank Fusion (RRF) reranking with constant k=60
    - Deduplicates documents while preserving highest RRF scores
    - Returns top 21 documents sorted by RRF score'''

    def __init__(self):
        super().__init__()
        self.k = 25  # Retrieve 25 documents per query
        self.rrf_constant = 60  # RRF constant from IR literature
        self.query_decomposer = ClaimQueryDecomposer()
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Step 1: Decompose claim into 3 diverse queries
        queries = self.query_decomposer(claim=claim)

        # Step 2: Parallel retrieval - retrieve k=25 documents per query (total 75 documents)
        docs_query1 = self.retrieve_k(queries.query1).passages
        docs_query2 = self.retrieve_k(queries.query2).passages
        docs_query3 = self.retrieve_k(queries.query3).passages

        # Step 3: Apply Reciprocal Rank Fusion (RRF) reranking
        # RRF score for each document = sum of 1/(rank + k) across all retrievals where it appears
        rrf_scores = {}

        # Process each retrieval list
        for rank, doc in enumerate(docs_query1):
            doc_key = doc  # Use the document itself as key
            rrf_score = 1.0 / (rank + self.rrf_constant)
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += rrf_score
            else:
                rrf_scores[doc_key] = rrf_score

        for rank, doc in enumerate(docs_query2):
            doc_key = doc
            rrf_score = 1.0 / (rank + self.rrf_constant)
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += rrf_score
            else:
                rrf_scores[doc_key] = rrf_score

        for rank, doc in enumerate(docs_query3):
            doc_key = doc
            rrf_score = 1.0 / (rank + self.rrf_constant)
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += rrf_score
            else:
                rrf_scores[doc_key] = rrf_score

        # Step 4: Sort documents by RRF score (descending) and deduplicate
        # The dictionary already handles deduplication with highest cumulative RRF scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5: Return top 21 documents
        top_21_docs = [doc for doc, score in sorted_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
