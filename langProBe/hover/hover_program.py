import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    STRATEGY
    - Uses parallel diversified retrieval with 3 complementary search strategies:
      1. Direct claim retrieval (k=21)
      2. Related entities/people/works retrieval (k=21)
      3. Background information/context retrieval (k=21)
    - Applies diversity-based reranking to select top 21 from all 63 retrieved documents
    - Eliminates information bottleneck from summarization steps'''

    def __init__(self):
        super().__init__()
        self.k = 21
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _diversified_rerank(self, all_docs, claim, top_k=21):
        """
        Select top_k most diverse documents from all_docs using maximal marginal relevance (MMR).
        Prioritizes documents that cover different entities/topics while maintaining relevance.

        Args:
            all_docs: List of document strings
            claim: Original claim string (for relevance scoring)
            top_k: Number of documents to select (default 21)

        Returns:
            List of top_k diverse documents
        """
        if len(all_docs) <= top_k:
            return all_docs

        # Remove exact duplicates while preserving order
        # Normalize by stripping whitespace and converting to lowercase
        unique_docs = []
        seen = set()
        for doc in all_docs:
            # Normalize: strip whitespace, lowercase, remove extra spaces
            doc_normalized = ' '.join(doc.strip().lower().split())
            if doc_normalized not in seen:
                seen.add(doc_normalized)
                unique_docs.append(doc)

        if len(unique_docs) <= top_k:
            return unique_docs

        # Create TF-IDF representations
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        # Include claim in vectorization to compute relevance
        all_texts = [claim] + unique_docs
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except:
            # Fallback: if TF-IDF fails, return first top_k unique docs
            return unique_docs[:top_k]

        # Claim vector is first, document vectors are rest
        claim_vector = tfidf_matrix[0:1]
        doc_vectors = tfidf_matrix[1:]

        # Compute relevance scores (similarity to claim)
        relevance_scores = cosine_similarity(doc_vectors, claim_vector).flatten()

        # Compute pairwise similarity between documents
        doc_similarity = cosine_similarity(doc_vectors)

        # MMR-based selection with lambda=0.5 (balanced diversity and relevance)
        selected_indices = []
        remaining_indices = list(range(len(unique_docs)))
        lambda_param = 0.5  # Balance between relevance and diversity

        # Start with most relevant document
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select documents with high relevance and low similarity to selected
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to any selected document)
                max_sim_to_selected = max(
                    doc_similarity[idx][selected_idx]
                    for selected_idx in selected_indices
                )

                # MMR score: high relevance, low similarity to selected
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                mmr_scores.append(mmr_score)

            # Select document with highest MMR score
            best_idx_in_remaining = int(np.argmax(mmr_scores))
            best_idx = remaining_indices[best_idx_in_remaining]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected documents in order of selection
        return [unique_docs[idx] for idx in selected_indices]

    def forward(self, claim):
        # HOP 1: Direct claim retrieval
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Related entities, people, or works
        hop2_query = f"related entities, people, or works mentioned in: {claim}"
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Background information and context
        hop3_query = f"background information and context about: {claim}"
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all retrieved documents (63 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Apply diversity-based reranking to select top 21
        diverse_docs = self._diversified_rerank(all_docs, claim, top_k=21)

        return dspy.Prediction(retrieved_docs=diverse_docs)
