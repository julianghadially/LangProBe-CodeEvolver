import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self, use_reranking=True, retrieve_k=30, final_k=7):
        """
        Initialize HoverMultiHop with optional reranking.

        Args:
            use_reranking: Whether to use LLM-based pairwise reranking (default: True)
            retrieve_k: Number of documents to retrieve per hop (default: 30 for wide retrieval)
            final_k: Number of documents to keep after reranking per hop (default: 7)
        """
        super().__init__()
        self.use_reranking = use_reranking
        self.retrieve_k = retrieve_k
        self.final_k = final_k

        # Query generation modules
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

        # Retrieval module with wide k
        self.retrieve = dspy.Retrieve(k=self.retrieve_k)

        # Reranking module (only if enabled)
        if self.use_reranking:
            from .hover_reranker import PairwiseReranker
            self.reranker = PairwiseReranker(k=self.final_k)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _deduplicate_by_title(self, docs: list[str]) -> list[str]:
        """
        Remove duplicate documents based on normalized title.
        Preserves first occurrence (earlier hops preferred).

        Args:
            docs: List of documents in "Title | Content" format

        Returns:
            List of unique documents (deduplicated by normalized title)
        """
        seen_titles = set()
        unique_docs = []

        for doc in docs:
            # Extract title (before " | ")
            title = doc.split(" | ")[0] if " | " in doc else doc
            # Normalize for comparison (consistent with evaluation metric)
            normalized_title = dspy.evaluate.normalize_text(title)

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

        return unique_docs

    def _retrieve_and_rerank(self, query: str, claim: str) -> list[str]:
        """
        Retrieve wide set of documents and rerank to top-k.

        This method implements the "Retrieve-Wide-Then-Rerank" strategy:
        1. Retrieve a large set of documents (e.g., k=30)
        2. Apply LLM-based pairwise reranking to select top documents (e.g., top-7)

        Args:
            query: The query string to retrieve documents for
            claim: The original claim (used for reranking relevance)

        Returns:
            List of top-k most relevant documents after reranking
        """
        # Wide retrieval
        raw_passages = self.retrieve(query).passages

        # Rerank if enabled
        if self.use_reranking:
            ranked_passages = self.reranker(
                claim=claim, passages=raw_passages
            ).ranked_passages
            return ranked_passages
        else:
            # Fallback: return first k documents (baseline behavior)
            return raw_passages[:self.final_k]

    def forward(self, claim):
        # HOP 1: Initial retrieval based on claim
        hop1_docs = self._retrieve_and_rerank(query=claim, claim=claim)
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Generate query based on first summary
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self._retrieve_and_rerank(query=hop2_query, claim=claim)
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Generate query based on both summaries
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self._retrieve_and_rerank(query=hop3_query, claim=claim)

        # Combine all documents and deduplicate
        all_docs = hop1_docs + hop2_docs + hop3_docs
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        # Ensure at most 21 documents (evaluation requirement)
        # Note: May have fewer than 21 after deduplication, which is acceptable
        final_docs = deduplicated_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
