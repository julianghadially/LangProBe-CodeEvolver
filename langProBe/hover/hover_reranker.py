import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class PairwiseComparisonSignature(dspy.Signature):
    """
    Given a claim and two documents, determine which document is more relevant
    for verifying or understanding the claim. Consider both topical relevance
    and informational value. The documents are formatted as "Title | Content snippet".
    """

    claim = dspy.InputField(desc="The claim to be verified or understood")
    doc1 = dspy.InputField(desc="First document with format 'Title | Content snippet'")
    doc2 = dspy.InputField(desc="Second document with format 'Title | Content snippet'")
    more_relevant_doc = dspy.OutputField(
        desc="Which document is more relevant: either '1' or '2'. "
             "If both are equally relevant, choose '1'."
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of why the chosen document is more relevant"
    )


class PairwiseReranker(LangProBeDSPyMetaProgram, dspy.Module):
    """
    Reranks documents using pairwise comparisons to identify the most relevant ones.
    Uses a bubble-sort-style tournament with early termination to find top-k documents.

    The reranking process:
    1. Takes a list of documents (e.g., 30 documents)
    2. Applies pairwise comparisons using LLM-based judgment
    3. Uses partial bubble sort to bubble up the top-k most relevant documents
    4. Returns the top-k documents in ranked order

    This approach is more expensive than embedding-based reranking but provides
    better semantic understanding of relevance through LLM reasoning.
    """

    def __init__(self, k=7):
        """
        Initialize the PairwiseReranker.

        Args:
            k: Number of top documents to return after reranking (default: 7)
        """
        super().__init__()
        self.k = k
        self.compare = dspy.ChainOfThought(PairwiseComparisonSignature)

    def _prepare_doc_snippet(self, doc: str, max_length: int = 200) -> str:
        """
        Extract title and content snippet from document string.

        Documents are formatted as "Title | Content". This method truncates
        the content to max_length characters to fit within token limits for
        pairwise comparison.

        Args:
            doc: Document string in "Title | Content" format
            max_length: Maximum length of content snippet (default: 200)

        Returns:
            Formatted string with truncated content: "Title | Content[0:max_length]..."
        """
        # Documents are formatted as "Title | Content"
        parts = doc.split(" | ", 1)
        if len(parts) == 2:
            title, content = parts
            # Truncate content to max_length chars
            if len(content) > max_length:
                content = content[:max_length] + "..."
            return f"{title} | {content}"
        # Fallback: if document doesn't have " | " separator, just truncate
        return doc[:max_length] + "..." if len(doc) > max_length else doc

    def _pairwise_compare(self, claim: str, doc1: str, doc2: str) -> int:
        """
        Compare two documents and return which is more relevant to the claim.

        Uses LLM-based Chain-of-Thought reasoning to determine relevance.
        The LLM is asked to choose between document 1 and document 2 based on
        which provides more relevant information for verifying or understanding
        the claim.

        Args:
            claim: The claim to verify
            doc1: First document (full text)
            doc2: Second document (full text)

        Returns:
            1 if doc1 is more relevant, 2 if doc2 is more relevant
        """
        snippet1 = self._prepare_doc_snippet(doc1)
        snippet2 = self._prepare_doc_snippet(doc2)

        result = self.compare(claim=claim, doc1=snippet1, doc2=snippet2)
        choice = result.more_relevant_doc.strip()

        # Parse response (handle various formats like "1", "2", "doc1", "doc2", "Document 1", etc.)
        if "2" in choice or "second" in choice.lower():
            return 2
        return 1  # Default to 1 if ambiguous or explicitly "1"

    def _bubble_sort_top_k(self, claim: str, docs: list[str]) -> list[str]:
        """
        Perform partial bubble sort to identify top-k documents.

        This is an optimized bubble sort that only performs k passes, which is
        sufficient to bubble the top-k most relevant documents to the beginning
        of the list.

        Algorithm:
        - Perform k passes through the list
        - In each pass, compare adjacent pairs from right to left
        - If the right document is more relevant, swap with left
        - After k passes, the top-k documents will be at positions [0:k]

        Complexity: O(n * k) where n is number of documents
        For n=30, k=7: ~210 comparisons worst case

        Args:
            claim: The claim to verify (used for relevance comparison)
            docs: List of documents to sort

        Returns:
            List of documents with top-k most relevant at the beginning
        """
        if len(docs) <= self.k:
            return docs

        docs_copy = docs.copy()
        n = len(docs_copy)

        # Only need k passes to get top-k elements
        for i in range(self.k):
            # Each pass bubbles one more element to the top
            # Compare from right to left, stopping at position i (already sorted)
            for j in range(n - 1, i, -1):
                comparison = self._pairwise_compare(
                    claim, docs_copy[j-1], docs_copy[j]
                )
                if comparison == 2:  # doc2 (right) is more relevant
                    # Swap: move more relevant doc leftward
                    docs_copy[j-1], docs_copy[j] = docs_copy[j], docs_copy[j-1]

        # Return top-k elements
        return docs_copy[:self.k]

    def forward(self, claim: str, passages: list[str]) -> dspy.Prediction:
        """
        Rerank passages and return top-k most relevant documents.

        This is the main entry point for the reranker. It takes a claim and
        a list of passages, applies pairwise reranking, and returns the top-k
        most relevant passages.

        Args:
            claim: The claim to verify
            passages: List of document strings to rerank

        Returns:
            dspy.Prediction with 'ranked_passages' field containing top-k docs
        """
        ranked_passages = self._bubble_sort_top_k(claim, passages)
        return dspy.Prediction(ranked_passages=ranked_passages)
