import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class GenerateMultipleQueries(dspy.Signature):
    """Generate 3 diverse search queries to verify a claim from different angles.

    The queries should cover:
    1. Key entities and their properties
    2. Relationships between entities
    3. Verification angles (temporal, causal, or contextual aspects)
    """
    claim = dspy.InputField(desc="The claim to verify")
    query1 = dspy.OutputField(desc="Query focusing on key entities and their properties")
    query2 = dspy.OutputField(desc="Query focusing on relationships between entities")
    query3 = dspy.OutputField(desc="Query focusing on verification angles (temporal, causal, or contextual)")

class RerankByCoverage(dspy.Signature):
    """Score and rerank documents based on coverage of different claim aspects.

    Evaluate each document for:
    - Entity mentions (which key entities from the claim appear)
    - Relationship coverage (how well it covers relationships between entities)
    - Factual relevance (how directly it addresses the claim's factual content)
    """
    claim = dspy.InputField(desc="The claim being verified")
    passages = dspy.InputField(desc="All retrieved passages to rerank")
    ranked_passages = dspy.OutputField(desc="Passages reranked by coverage score, with explanation of coverage aspects")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using parallel multi-query retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20  # Retrieve 20 documents per query
        self.generate_queries = dspy.ChainOfThought(GenerateMultipleQueries)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_by_coverage = dspy.ChainOfThought(RerankByCoverage)

    def forward(self, claim):
        # STEP 1: Generate 3 diverse queries covering different aspects of the claim
        queries_result = self.generate_queries(claim=claim)
        query1 = queries_result.query1
        query2 = queries_result.query2
        query3 = queries_result.query3

        # STEP 2: Parallel retrieval - retrieve k=20 documents for each query
        docs1 = self.retrieve_k(query1).passages
        docs2 = self.retrieve_k(query2).passages
        docs3 = self.retrieve_k(query3).passages

        # Combine all retrieved passages (up to ~60 documents)
        all_passages = docs1 + docs2 + docs3

        # Remove duplicates while preserving order
        seen = set()
        unique_passages = []
        for passage in all_passages:
            # Use passage content as key for deduplication
            passage_key = passage if isinstance(passage, str) else str(passage)
            if passage_key not in seen:
                seen.add(passage_key)
                unique_passages.append(passage)

        # STEP 3: Coverage-based reranking
        rerank_result = self.rerank_by_coverage(
            claim=claim,
            passages=unique_passages
        )

        # STEP 4: Select top 21 documents from reranked results
        # The ranked_passages output should be a string or list; parse accordingly
        ranked_passages_output = rerank_result.ranked_passages

        # Handle the reranked output - assume it returns passages in ranked order
        if isinstance(ranked_passages_output, str):
            # If it's a string description, use original unique passages (fallback)
            final_docs = unique_passages[:21]
        elif isinstance(ranked_passages_output, list):
            final_docs = ranked_passages_output[:21]
        else:
            # Fallback to first 21 unique passages
            final_docs = unique_passages[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
