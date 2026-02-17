import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class SimpleQueryGeneratorSignature(dspy.Signature):
    """Generate diverse search queries to explore different aspects of a claim.
    Create queries that approach the claim from different angles to maximize entity and topic coverage."""

    claim = dspy.InputField(desc="The factual claim to verify")
    hop_number = dspy.InputField(desc="Which retrieval hop this is (1, 2, or 3)")
    previous_query = dspy.InputField(desc="The query used in the previous hop (empty for hop 1)")

    query = dspy.OutputField(desc="A search query that explores a different angle of the claim to retrieve diverse documents")
    rationale = dspy.OutputField(desc="Brief explanation of what aspect this query targets")


class DiversityRerankerSignature(dspy.Signature):
    """Score documents based on how many distinct entities and topics they cover relative to the claim.
    Prioritize documents that contribute unique information and cover different aspects of the verification task."""

    claim = dspy.InputField(desc="The factual claim being verified")
    document_title = dspy.InputField(desc="Title of the document to score")
    document_content = dspy.InputField(desc="Content of the document (first 500 chars)")
    already_covered_topics = dspy.InputField(desc="Topics and entities already covered by higher-ranked documents")

    diversity_score = dspy.OutputField(desc="Score from 1-10: how much unique/novel information this document provides")
    unique_topics = dspy.OutputField(desc="List of distinct entities/topics this document uniquely covers")
    rationale = dspy.OutputField(desc="Brief explanation of the diversity score")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval system with diversity-boosting pipeline.
    Retrieves large candidate set (105 documents), eliminates duplicates via title normalization,
    then uses diversity reranking to select top 21 documents with maximum entity/topic coverage.

    EVALUATION
    - Retrieves 105 raw documents (35 per hop × 3 hops)
    - Eliminates duplicates through aggressive title normalization
    - Reranks remaining unique documents by diversity score
    - Returns exactly 21 most diverse documents
    - Maximizes entity coverage without complex coverage analysis'''

    def __init__(self):
        super().__init__()
        self.k = 35  # Increased from 7 to 35 per hop
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.final_k = 21  # Final number of documents to return

        # Simplified query generation
        self.query_generator_hop2 = dspy.ChainOfThought(SimpleQueryGeneratorSignature)
        self.query_generator_hop3 = dspy.ChainOfThought(SimpleQueryGeneratorSignature)

        # Diversity reranking
        self.diversity_reranker = dspy.ChainOfThought(DiversityRerankerSignature)

    def _normalize_title(self, title: str) -> str:
        """Normalize title for duplicate detection.
        Remove special characters, convert to lowercase, strip whitespace."""
        import re
        normalized = title.lower().strip()
        # Remove special characters and extra whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _extract_title_and_content(self, passage: str) -> tuple[str, str]:
        """Extract title and content from passage in 'title | content' format"""
        parts = passage.split(" | ", 1)
        title = parts[0] if len(parts) > 0 else ""
        content = parts[1] if len(parts) > 1 else ""
        return title, content

    def _deduplicate_documents(self, passages: list[str]) -> list[str]:
        """Remove duplicate documents by normalized title.
        Keeps first occurrence of each unique title."""
        seen_titles = set()
        unique_passages = []

        for passage in passages:
            title, _ = self._extract_title_and_content(passage)
            normalized = self._normalize_title(title)

            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_passages.append(passage)

        return unique_passages

    def _rerank_by_diversity(self, claim: str, passages: list[str]) -> list[str]:
        """Rerank documents by diversity score to maximize entity/topic coverage.
        Returns top 21 most diverse documents."""
        scored_passages = []
        covered_topics = []

        # Score each document based on unique information it provides
        for passage in passages:
            title, content = self._extract_title_and_content(passage)
            # Truncate content to first 500 chars for efficiency
            content_preview = content[:500]

            # Get diversity score
            diversity_output = self.diversity_reranker(
                claim=claim,
                document_title=title,
                document_content=content_preview,
                already_covered_topics=", ".join(covered_topics) if covered_topics else "None yet"
            )

            # Parse score (handle potential string output)
            try:
                score = float(diversity_output.diversity_score)
            except (ValueError, TypeError):
                # If score is not a number, try to extract first number from string
                import re
                score_match = re.search(r'\d+', str(diversity_output.diversity_score))
                score = float(score_match.group()) if score_match else 5.0

            scored_passages.append((score, passage))

            # Add unique topics to covered list
            if hasattr(diversity_output, 'unique_topics'):
                covered_topics.append(str(diversity_output.unique_topics))

        # Sort by score (descending) and take top 21
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        return [passage for _, passage in scored_passages[:self.final_k]]

    def forward(self, claim):
        # HOP 1: Direct claim-based retrieval (35 documents)
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Diversified query (35 documents)
        hop2_query_output = self.query_generator_hop2(
            claim=claim,
            hop_number="2",
            previous_query=claim
        )
        hop2_docs = self.retrieve_k(hop2_query_output.query).passages

        # HOP 3: Further diversified query (35 documents)
        hop3_query_output = self.query_generator_hop3(
            claim=claim,
            hop_number="3",
            previous_query=hop2_query_output.query
        )
        hop3_docs = self.retrieve_k(hop3_query_output.query).passages

        # Combine all 105 documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # STEP 1: Aggressive duplicate elimination by title normalization
        unique_docs = self._deduplicate_documents(all_docs)

        # STEP 2: Diversity reranking to select top 21 documents
        final_docs = self._rerank_by_diversity(claim, unique_docs)

        # Return exactly 21 diverse documents
        return dspy.Prediction(retrieved_docs=final_docs)
