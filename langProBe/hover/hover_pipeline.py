import dspy
from typing import Dict, List, Tuple
from collections import defaultdict
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

    def _extract_title(self, doc: str) -> str:
        """Extract the title from a document in 'title | content' format."""
        return doc.split(" | ")[0] if " | " in doc else doc

    def _compute_position_score(self, position: int, total_docs: int) -> float:
        """Compute position-based score. Earlier positions get higher scores."""
        # Linear decay: first position gets 1.0, last position gets 0.0
        return 1.0 - (position / max(total_docs, 1))

    def _score_documents(self, queries_docs: List[List[str]]) -> List[str]:
        """
        Score documents based on query-intersection and position.

        Args:
            queries_docs: List of document lists, one per query

        Returns:
            List of top 21 unique documents sorted by combined score
        """
        # Track scores for each document title
        doc_scores: Dict[str, float] = defaultdict(float)
        # Track which queries each document appears in
        doc_query_appearances: Dict[str, set] = defaultdict(set)
        # Track the best (original) document string for each title
        doc_strings: Dict[str, str] = {}

        # Process each query's results
        for query_idx, docs in enumerate(queries_docs):
            for position, doc in enumerate(docs):
                title = self._extract_title(doc)

                # Store the first occurrence of this document
                if title not in doc_strings:
                    doc_strings[title] = doc

                # Track which query this document appeared in
                doc_query_appearances[title].add(query_idx)

                # Add position-based score
                position_score = self._compute_position_score(position, len(docs))
                doc_scores[title] += position_score

        # Apply cross-query frequency bonus
        num_queries = len(queries_docs)
        for title in doc_scores:
            num_appearances = len(doc_query_appearances[title])
            # Bonus multiplier: documents appearing in multiple queries get exponential boost
            # 1 query: 1.0x, 2 queries: 2.0x, 3 queries: 3.0x
            cross_query_bonus = num_appearances
            doc_scores[title] *= cross_query_bonus

        # Sort documents by score (descending) and take top 21
        sorted_titles = sorted(doc_scores.keys(), key=lambda t: doc_scores[t], reverse=True)
        top_titles = sorted_titles[:21]

        # Return the original document strings
        return [doc_strings[title] for title in top_titles]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Get the raw multi-hop retrieval results
            result = self.program(claim=claim)

            # Extract documents from each hop (3 queries: original claim, hop2 query, hop3 query)
            # The program returns all docs concatenated, but we need them separated by query
            # Each hop retrieves k=7 documents
            k = self.program.k
            all_docs = result.retrieved_docs

            # Split documents by query/hop
            hop1_docs = all_docs[:k]
            hop2_docs = all_docs[k:2*k]
            hop3_docs = all_docs[2*k:3*k]

            queries_docs = [hop1_docs, hop2_docs, hop3_docs]

            # Apply query-intersection scoring and reranking
            reranked_docs = self._score_documents(queries_docs)

            return dspy.Prediction(retrieved_docs=reranked_docs)
