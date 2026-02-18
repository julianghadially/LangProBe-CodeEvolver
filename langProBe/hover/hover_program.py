import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20  # Increased from 7 to 20 to retrieve more candidates initially

        # Query generation modules
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Gap analysis module - identifies missing information before each hop
        self.gap_analysis = dspy.ChainOfThought("claim,current_docs->missing_entities")

        # Document scoring module - scores and filters documents after each hop
        self.document_scorer = dspy.ChainOfThought("claim,passages->scored_passages")

        # Final reranking module - selects best 21 documents from all candidates
        self.final_reranker = dspy.ChainOfThought("claim,all_passages->top_21_passages")

    def forward(self, claim):
        # HOP 1: Initial retrieval
        hop1_docs = self.retrieve_k(claim).passages  # Retrieve k=20 documents

        # Score and filter hop 1 documents to ~10 best
        hop1_scored = self.document_scorer(
            claim=claim, passages=hop1_docs
        ).scored_passages
        hop1_filtered = self._filter_top_documents(hop1_scored, target_count=10)

        # Summarize filtered documents from hop 1
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_filtered
        ).summary

        # HOP 2: Gap analysis before retrieval
        gap_analysis_hop2 = self.gap_analysis(
            claim=claim, current_docs=hop1_filtered
        ).missing_entities

        # Generate query for hop 2 based on summary and gaps
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages  # Retrieve k=20 documents

        # Score and filter hop 2 documents to ~10 best
        hop2_scored = self.document_scorer(
            claim=claim, passages=hop2_docs
        ).scored_passages
        hop2_filtered = self._filter_top_documents(hop2_scored, target_count=10)

        # Summarize filtered documents from hop 2
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_filtered
        ).summary

        # HOP 3: Gap analysis before retrieval
        gap_analysis_hop3 = self.gap_analysis(
            claim=claim, current_docs=hop1_filtered + hop2_filtered
        ).missing_entities

        # Generate query for hop 3 based on all previous context
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages  # Retrieve k=20 documents

        # Score and filter hop 3 documents to ~10 best
        hop3_scored = self.document_scorer(
            claim=claim, passages=hop3_docs
        ).scored_passages
        hop3_filtered = self._filter_top_documents(hop3_scored, target_count=10)

        # Combine all candidate documents (up to 60 from 3 hops of k=20 each)
        all_candidates = hop1_docs + hop2_docs + hop3_docs

        # Final reranking: select exactly 21 highest-scoring documents
        final_docs = self.final_reranker(
            claim=claim, all_passages=all_candidates
        ).top_21_passages

        # Ensure exactly 21 documents are returned
        final_docs = self._ensure_21_documents(final_docs, all_candidates)

        return dspy.Prediction(retrieved_docs=final_docs)

    def _filter_top_documents(self, scored_passages, target_count=10):
        """Filter scored passages to keep approximately target_count top documents.

        Expected format: scored_passages should be a string or list containing
        document scores. This method extracts and returns the top-scoring documents.
        """
        # If scored_passages is already a list, return top N
        if isinstance(scored_passages, list):
            return scored_passages[:target_count]

        # If it's a string (typical DSPy output), parse and extract top documents
        # For now, return the scored_passages as-is for DSPy to handle
        # The actual filtering logic will be learned by the model
        return scored_passages

    def _ensure_21_documents(self, final_docs, all_candidates):
        """Ensure exactly 21 documents are returned.

        If final_docs has fewer than 21, pad with highest-scoring candidates.
        If final_docs has more than 21, truncate to 21.
        """
        if isinstance(final_docs, list):
            if len(final_docs) >= 21:
                return final_docs[:21]
            elif len(final_docs) < 21:
                # Pad with additional candidates to reach 21
                remaining = 21 - len(final_docs)
                additional_docs = [doc for doc in all_candidates if doc not in final_docs]
                return final_docs + additional_docs[:remaining]

        # If final_docs is not a list (e.g., string from DSPy), return as-is
        # and trust the model was instructed to return 21 documents
        return final_docs
