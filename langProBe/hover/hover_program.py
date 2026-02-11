import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateSubQueriesHop2(dspy.Signature):
    """Analyze the claim and first-hop documents to identify information gaps.
    Generate 3 specific, diverse sub-queries targeting missing entities, relationships,
    dates, or facts needed to verify the claim."""

    claim = dspy.InputField(desc="The claim to verify through multi-hop fact retrieval")
    retrieved_context = dspy.InputField(
        desc="Documents retrieved in hop 1, formatted with titles and content"
    )
    sub_queries: list[str] = dspy.OutputField(
        desc="Exactly 3 specific, diverse search queries targeting missing information. "
             "Focus on: (1) entities mentioned but not retrieved, (2) relationships, "
             "(3) temporal information, (4) related concepts. Make each query independent."
    )


class GenerateSubQueriesHop3(dspy.Signature):
    """Perform final retrieval by analyzing all documents so far.
    Generate 4 targeted sub-queries to fill remaining gaps and verify entity connections."""

    claim = dspy.InputField(desc="The claim to verify through multi-hop fact retrieval")
    retrieved_context = dspy.InputField(
        desc="All documents from hop 1 and hop 2, formatted with titles"
    )
    sub_queries: list[str] = dspy.OutputField(
        desc="Exactly 4 specific queries for final verification. "
             "Prioritize: (1) entity relationship verification, (2) missing context, "
             "(3) temporal sequences, (4) alternative entity names/aliases."
    )


class ScoreDocumentRelevance(dspy.Signature):
    """Evaluate how relevant a document is for verifying a specific claim.
    Consider: factual overlap, entity coverage, temporal information, and comparative value."""

    claim = dspy.InputField(desc="The claim to verify")
    document = dspy.InputField(desc="Document text in 'title | content' format")
    relevance_score: int = dspy.OutputField(
        desc="Relevance score from 1-10, where 10 is highly relevant and 1 is irrelevant"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this score was assigned, highlighting key factors"
    )


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()

        # Hop 1: Single query, k=7
        self.retrieve_hop1 = dspy.Retrieve(k=7)

        # Hop 2: Generate 3 sub-queries, retrieve k=2 each (total 6 docs)
        self.generate_subqueries_hop2 = dspy.ChainOfThought(GenerateSubQueriesHop2)
        self.retrieve_hop2 = dspy.Retrieve(k=2)

        # Hop 3: Generate 4 sub-queries, retrieve k=2 each (total 8 docs)
        self.generate_subqueries_hop3 = dspy.ChainOfThought(GenerateSubQueriesHop3)
        self.retrieve_hop3 = dspy.Retrieve(k=2)

        # Reranking: Score document relevance
        self.score_relevance = dspy.ChainOfThought(ScoreDocumentRelevance)

        # Query count constants
        self.hop2_num_queries = 3
        self.hop3_num_queries = 4

    def forward(self, claim):
        # ===== HOP 1: Direct retrieval on claim =====
        hop1_docs = self.retrieve_hop1(claim).passages  # 7 documents

        # Format for context
        hop1_context_str = self._format_docs_for_context(hop1_docs)

        # ===== HOP 2: Multi-query fan-out (3 queries × k=2 = 6 docs) =====
        hop2_subqueries_pred = self.generate_subqueries_hop2(
            claim=claim,
            retrieved_context=hop1_context_str
        )
        hop2_subqueries = self._normalize_query_list(
            hop2_subqueries_pred.sub_queries,
            expected_count=self.hop2_num_queries
        )

        # Execute retrievals for each sub-query
        hop2_docs = []
        for query in hop2_subqueries:
            retrieved = self.retrieve_hop2(query).passages  # k=2 per query
            hop2_docs.extend(retrieved)

        # Combine contexts for hop 3
        all_context_so_far = hop1_docs + hop2_docs
        all_context_str = self._format_docs_for_context(all_context_so_far)

        # ===== HOP 3: Multi-query fan-out (4 queries × k=2 = 8 docs) =====
        hop3_subqueries_pred = self.generate_subqueries_hop3(
            claim=claim,
            retrieved_context=all_context_str
        )
        hop3_subqueries = self._normalize_query_list(
            hop3_subqueries_pred.sub_queries,
            expected_count=self.hop3_num_queries
        )

        # Execute retrievals for each sub-query
        hop3_docs = []
        for query in hop3_subqueries:
            retrieved = self.retrieve_hop3(query).passages  # k=2 per query
            hop3_docs.extend(retrieved)

        # ===== FINAL: Combine all documents (7 + 6 + 8 = 21) =====
        all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs

        # ===== RERANKING: Score and deduplicate documents =====
        reranked_docs = self._rerank_with_diversity(claim, all_retrieved_docs)

        return dspy.Prediction(retrieved_docs=reranked_docs)

    def _format_docs_for_context(self, docs: list[str]) -> str:
        """Format documents into numbered, readable context string.
        Documents are in 'title | content' format."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[{i}] {doc}")
        return "\n\n".join(formatted)

    def _normalize_query_list(self, queries: list[str], expected_count: int) -> list[str]:
        """Ensure query list has exactly expected_count queries.

        Handles edge cases:
        - If fewer: Pad by repeating the last query
        - If more: Truncate to expected_count
        - If empty: Repeat the claim as fallback queries
        """
        if not queries:
            # Fallback: shouldn't happen with ChainOfThought
            return [f"fallback_query_{i}" for i in range(expected_count)]

        if len(queries) < expected_count:
            # Pad by repeating last query
            padding = expected_count - len(queries)
            return queries + [queries[-1]] * padding

        # Truncate if too many
        return queries[:expected_count]

    def _rerank_with_diversity(self, claim: str, documents: list[str]) -> list[str]:
        """Rerank documents by relevance with diversity-aware deduplication.

        Steps:
        1. Score each document for relevance (1-10)
        2. Group by normalized title (deduplication)
        3. Keep highest-scored instance of each unique title
        4. Fill remaining slots (up to 21) with next-highest scored documents

        Args:
            claim: The claim to verify
            documents: List of documents in 'title | content' format

        Returns:
            Reranked and deduplicated document list (up to 21 documents)
        """
        # Score all documents
        scored_docs = []
        for doc in documents:
            try:
                prediction = self.score_relevance(claim=claim, document=doc)
                score = prediction.relevance_score

                # Ensure score is an integer in valid range
                if isinstance(score, str):
                    score = int(score)
                score = max(1, min(10, score))  # Clamp to [1, 10]

                scored_docs.append({
                    'document': doc,
                    'score': score,
                    'reasoning': prediction.reasoning
                })
            except Exception as e:
                # Fallback: assign neutral score if scoring fails
                scored_docs.append({
                    'document': doc,
                    'score': 5,
                    'reasoning': f"Scoring failed: {str(e)}"
                })

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # Extract normalized title from 'title | content' format
        def get_normalized_title(doc_text: str) -> str:
            """Extract and normalize title from document text."""
            if ' | ' in doc_text:
                title = doc_text.split(' | ')[0].strip()
            else:
                # Fallback: use first 50 chars as title
                title = doc_text[:50].strip()

            # Normalize: lowercase, remove extra whitespace
            return ' '.join(title.lower().split())

        # Diversity-aware selection
        seen_titles = set()
        selected_docs = []
        overflow_docs = []

        for item in scored_docs:
            doc = item['document']
            title = get_normalized_title(doc)

            if title not in seen_titles:
                # New unique title: always add
                seen_titles.add(title)
                selected_docs.append(doc)
            else:
                # Duplicate title: save for potential backfill
                overflow_docs.append(doc)

        # Fill remaining slots with overflow (allows some duplication if needed)
        max_docs = 21
        remaining_slots = max_docs - len(selected_docs)
        if remaining_slots > 0:
            selected_docs.extend(overflow_docs[:remaining_slots])

        return selected_docs


