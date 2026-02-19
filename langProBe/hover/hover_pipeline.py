import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class QueryVariationSignature(dspy.Signature):
    """Generate 2-3 diverse query variations that rephrase the same information need from different angles."""

    query: str = dspy.InputField(desc="the original query to expand")
    variations: list[str] = dspy.OutputField(desc="2-3 diverse query variations rephrasing the same information need from different angles")


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked document lists (each list is ordered by relevance)
        k: RRF constant (default 60)

    Returns:
        Merged and reranked list of documents
    """
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            # RRF score: 1 / (k + rank)
            # rank is 0-indexed, so we use rank + 1
            score = 1.0 / (k + rank + 1)
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += score

    # Sort by RRF score (descending)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k_retrieve = 30  # Retrieve 30 docs per query variation
        self.k_final = 7      # Keep top 7 after RRF
        self.rrf_k = 60       # RRF constant

        # Query generation modules
        self.query_expander = dspy.Predict(QueryVariationSignature)
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k_retrieve)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def retrieve_with_query_expansion(self, base_query: str) -> list[str]:
        """
        Retrieve documents using query expansion and RRF reranking.

        Args:
            base_query: The original query

        Returns:
            Top-k documents after RRF reranking
        """
        # Generate query variations
        variations_result = self.query_expander(query=base_query)
        query_variations = variations_result.variations

        # Ensure we have 2-3 variations, include base query as fallback
        if not isinstance(query_variations, list):
            query_variations = [query_variations]

        # Use the original query plus the variations (up to 3 total)
        all_queries = [base_query] + query_variations[:2]

        # Retrieve documents for each query variation
        ranked_lists = []
        for query in all_queries:
            docs = self.retrieve_k(query).passages
            ranked_lists.append(docs)

        # Apply RRF to merge results
        merged_docs = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        # Return top-k documents
        return merged_docs[:self.k_final]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Direct retrieval from claim with query expansion
            hop1_docs = self.retrieve_with_query_expansion(claim)
            summary_1 = self.summarize1(
                claim=claim, passages=hop1_docs
            ).summary

            # HOP 2: Generate query, then retrieve with expansion
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = self.retrieve_with_query_expansion(hop2_query)
            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs
            ).summary

            # HOP 3: Generate final query, then retrieve with expansion
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query
            hop3_docs = self.retrieve_with_query_expansion(hop3_query)

            # Combine all documents (7 + 7 + 7 = 21 total)
            return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
