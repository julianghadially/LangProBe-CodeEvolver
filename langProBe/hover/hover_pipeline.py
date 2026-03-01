import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ListwiseReranker(dspy.Signature):
    """Evaluate and rank documents by their relevance to a claim using chain-of-thought reasoning.

    Analyze each document's relevance to the claim, considering:
    1. How directly the document addresses the claim
    2. Whether the document provides supporting or contradicting evidence
    3. The quality and specificity of information in the document

    Output a ranked list of document indices (0-based) in descending order of relevance."""

    claim: str = dspy.InputField(desc="The claim to verify")
    passages: list[str] = dspy.InputField(desc="List of document passages to evaluate and rank")
    ranked_indices: list[int] = dspy.OutputField(desc="Ranked list of document indices (0-based), from most to least relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        # Use k=21 to retrieve 63 total documents (21 per hop across 3 hops)
        self.program = HoverMultiHop(k=21)
        # Initialize the listwise reranker with chain-of-thought reasoning
        self.reranker = dspy.ChainOfThought(ListwiseReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Retrieve 63 documents total (21 per hop)
            result = self.program(claim=claim)
            all_docs = result.retrieved_docs

            # Apply sliding-window listwise reranking
            reranked_docs = self._sliding_window_rerank(claim, all_docs)

            # Return top 21 documents
            return dspy.Prediction(retrieved_docs=reranked_docs[:21])

    def _sliding_window_rerank(self, claim, documents, window_size=30, top_k=21):
        """Apply sliding-window listwise reranking to select top documents.

        Args:
            claim: The claim to verify
            documents: List of all retrieved documents (63 total)
            window_size: Size of each sliding window (default: 30)
            top_k: Number of top documents to return (default: 21)

        Returns:
            List of top-k reranked documents
        """
        if len(documents) <= top_k:
            return documents

        # Initialize scores for all documents
        doc_scores = [0.0] * len(documents)
        doc_count = [0] * len(documents)  # Track how many windows each doc appears in

        # Create overlapping windows
        # For 63 docs with window_size=30, we create windows at positions 0, 15, 30, etc.
        stride = window_size // 2
        windows = []
        for start_idx in range(0, len(documents), stride):
            end_idx = min(start_idx + window_size, len(documents))
            if end_idx - start_idx < 10:  # Skip very small windows at the end
                break
            windows.append((start_idx, end_idx))

        # Process each window
        for start_idx, end_idx in windows:
            window_docs = documents[start_idx:end_idx]

            # Get rankings for this window using the LLM reranker
            try:
                rerank_result = self.reranker(claim=claim, passages=window_docs)
                ranked_indices = rerank_result.ranked_indices

                # Assign scores based on rank position (higher rank = higher score)
                # Use reciprocal rank scoring: 1/rank
                for rank_position, local_idx in enumerate(ranked_indices):
                    if isinstance(local_idx, int) and 0 <= local_idx < len(window_docs):
                        global_idx = start_idx + local_idx
                        # Score is inversely proportional to rank (1-indexed)
                        score = 1.0 / (rank_position + 1)
                        doc_scores[global_idx] += score
                        doc_count[global_idx] += 1
            except Exception as e:
                # If reranking fails for a window, assign uniform scores
                for i in range(len(window_docs)):
                    global_idx = start_idx + i
                    doc_scores[global_idx] += 1.0 / len(window_docs)
                    doc_count[global_idx] += 1

        # Average scores across windows
        averaged_scores = [
            score / max(count, 1) for score, count in zip(doc_scores, doc_count)
        ]

        # Sort documents by averaged score and return top_k
        doc_score_pairs = list(enumerate(averaged_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Extract top-k documents in their new order
        reranked_docs = [documents[idx] for idx, _ in doc_score_pairs[:top_k]]

        return reranked_docs
