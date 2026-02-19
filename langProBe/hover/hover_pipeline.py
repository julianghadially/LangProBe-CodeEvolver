import dspy
import numpy as np
from typing import List
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class MMRReranker(dspy.Module):
    """Maximal Marginal Relevance reranker for diversifying document selection.

    Implements MMR formula: score = λ * relevance(doc, claim) - (1-λ) * max_similarity(doc, selected_docs)
    to balance relevance to the claim with diversity from already-selected documents.
    """

    def __init__(self, lambda_param: float = 0.7, final_k: int = 21):
        super().__init__()
        self.lambda_param = lambda_param
        self.final_k = final_k
        # Use DSPy's embedding capabilities for semantic similarity

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using simple token overlap.

        For production systems, this could be replaced with embedding-based similarity.
        Here we use a normalized token overlap metric as a proxy.
        """
        # Extract content from passages (format: "title | content")
        content1 = text1.split(" | ")[-1] if " | " in text1 else text1
        content2 = text2.split(" | ")[-1] if " | " in text2 else text2

        # Tokenize and compute Jaccard similarity
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def _compute_relevance(self, doc: str, claim: str) -> float:
        """Compute relevance of document to claim using semantic similarity."""
        return self._compute_similarity(doc, claim)

    def _compute_max_similarity(self, doc: str, selected_docs: List[str]) -> float:
        """Compute maximum similarity between doc and any selected document."""
        if not selected_docs:
            return 0.0

        similarities = [self._compute_similarity(doc, sel_doc) for sel_doc in selected_docs]
        return max(similarities)

    def forward(self, claim: str, passages: List[str]) -> dspy.Prediction:
        """Apply MMR to select final_k diverse and relevant documents.

        Args:
            claim: The claim to verify
            passages: All retrieved passages (e.g., 105 documents from 3 hops)

        Returns:
            dspy.Prediction with reranked_docs containing the final_k selected documents
        """
        if len(passages) <= self.final_k:
            return dspy.Prediction(reranked_docs=passages)

        # Precompute relevance scores for all documents
        relevance_scores = {i: self._compute_relevance(doc, claim)
                          for i, doc in enumerate(passages)}

        selected_indices = []
        selected_docs = []
        remaining_indices = set(range(len(passages)))

        # Iteratively select documents using MMR
        for _ in range(self.final_k):
            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                doc = passages[idx]
                relevance = relevance_scores[idx]

                # Compute diversity term (max similarity to selected docs)
                max_sim = self._compute_max_similarity(doc, selected_docs)

                # MMR score: λ * relevance - (1-λ) * max_similarity
                mmr_score = (self.lambda_param * relevance -
                           (1 - self.lambda_param) * max_sim)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Select the document with highest MMR score
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_docs.append(passages[best_idx])
                remaining_indices.remove(best_idx)

        return dspy.Prediction(reranked_docs=selected_docs)


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with MMR reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Retrieves k=35 documents per hop (105 total across 3 hops)
    - Applies Maximal Marginal Relevance (MMR) reranking to select final 21 documents
    - MMR balances relevance to claim with diversity from selected documents (λ=0.7)
    '''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k_per_hop = 35
        self.final_k = 21
        self.mmr_reranker = MMRReranker(lambda_param=0.7, final_k=self.final_k)
        # Create internal retrieval program with modified k
        self.program = HoverMultiHop()
        # Override the k value for retrieval
        self.program.k = self.k_per_hop
        self.program.retrieve_k = dspy.Retrieve(k=self.k_per_hop)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Retrieve 35 documents per hop (105 total)
            retrieval_result = self.program(claim=claim)
            all_retrieved_docs = retrieval_result.retrieved_docs

            # Apply MMR reranking to select final 21 diverse and relevant documents
            mmr_result = self.mmr_reranker(claim=claim, passages=all_retrieved_docs)

            # Return the reranked documents
            return dspy.Prediction(retrieved_docs=mmr_result.reranked_docs)
