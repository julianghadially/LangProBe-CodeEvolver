import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractKeyEntities(dspy.Signature):
    """Extract named entities (people, places, organizations) from a claim.
    Focus on entities that would be useful for fact-checking and verification.
    Return up to 3 most important entities."""

    claim = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(
        desc="List of up to 3 key named entities (people, places, organizations) "
             "mentioned in the claim. Return entities that are most relevant for "
             "fact-checking. Each entity should be a proper noun or named entity."
    )


class GenerateEntityQueries(dspy.Signature):
    """Generate focused search queries for each entity to retrieve relevant context.
    Each query should target information about the specific entity that would help verify the claim."""

    claim = dspy.InputField(desc="The claim to verify through multi-hop fact retrieval")
    entity = dspy.InputField(desc="A specific named entity to search for")
    retrieved_context = dspy.InputField(desc="Documents already retrieved, formatted with titles")
    search_query: str = dspy.OutputField(
        desc="A specific search query focused on this entity in the context of the claim. "
             "The query should retrieve information about the entity's relationship to the claim."
    )


class GenerateSubQueriesHop3(dspy.Signature):
    """Analyze the claim and all retrieved documents to identify remaining information gaps.
    Generate 2-3 targeted queries to fill gaps, focusing on:
    1. Missing entities not yet retrieved
    2. Entity relationships and connections
    3. Temporal information or context
    Adapt the number of queries based on how much information is still missing."""

    claim = dspy.InputField(desc="The claim to verify through multi-hop fact retrieval")
    retrieved_context = dspy.InputField(
        desc="All documents from hop 1 and hop 2, formatted with titles. "
             "Analyze what entities and relationships are still missing."
    )
    sub_queries: list[str] = dspy.OutputField(
        desc="Between 2 and 3 specific queries to fill remaining gaps. "
             "Each query should target missing entities, relationships, or context. "
             "If most information is covered, generate 2 queries. "
             "If significant gaps remain, generate 3 queries."
    )


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()

        # Hop 1: Single query, k=7
        self.retrieve_hop1 = dspy.Retrieve(k=7)

        # Hop 2: Entity extraction and entity-based queries
        self.extract_entities = dspy.ChainOfThought(ExtractKeyEntities)
        self.generate_entity_query = dspy.ChainOfThought(GenerateEntityQueries)
        self.retrieve_hop2 = dspy.Retrieve(k=2)
        self.max_entities = 3

        # Hop 3: Gap-aware query generation
        self.generate_subqueries_hop3 = dspy.ChainOfThought(GenerateSubQueriesHop3)
        self.retrieve_hop3 = dspy.Retrieve(k=2)
        self.hop3_num_queries = 3  # Changed from 4 to 3

    def forward(self, claim):
        # ===== HOP 1: Direct retrieval on claim =====
        hop1_docs = self.retrieve_hop1(claim).passages  # 7 documents
        hop1_context_str = self._format_docs_for_context(hop1_docs)

        # ===== HOP 2: Entity-based retrieval =====
        # Extract key entities from the claim
        entities_pred = self.extract_entities(claim=claim)
        entities = self._normalize_entity_list(
            entities_pred.entities,
            max_entities=self.max_entities
        )

        # Generate and execute one query per entity
        hop2_docs = []
        for entity in entities:
            query_pred = self.generate_entity_query(
                claim=claim,
                entity=entity,
                retrieved_context=hop1_context_str
            )
            retrieved = self.retrieve_hop2(query_pred.search_query).passages
            hop2_docs.extend(retrieved)

        # Deduplicate before Hop 3 to maximize unique coverage
        all_context_so_far = self._deduplicate_by_title(hop1_docs + hop2_docs)
        all_context_str = self._format_docs_for_context(all_context_so_far)

        # ===== HOP 3: Gap-aware retrieval =====
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
            retrieved = self.retrieve_hop3(query).passages
            hop3_docs.extend(retrieved)

        # ===== FINAL: Deduplicate and limit to exactly 21 documents =====
        all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = self._deduplicate_by_title(all_retrieved_docs)
        final_docs = unique_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)

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

    def _normalize_entity_list(self, entities: list[str], max_entities: int) -> list[str]:
        """Ensure entity list has at most max_entities items.

        Handles edge cases:
        - If empty: Return single fallback entity
        - If too many: Truncate to max_entities
        - Cleans whitespace and removes duplicates
        """
        if not entities:
            return ["main claim"]

        # Remove whitespace, empty strings, and duplicates while preserving order
        seen = set()
        cleaned = []
        for entity in entities:
            entity = entity.strip()
            if entity and entity not in seen:
                seen.add(entity)
                cleaned.append(entity)

        # Truncate to max_entities
        return cleaned[:max_entities]

    def _deduplicate_by_title(self, docs: list[str]) -> list[str]:
        """Deduplicate documents by title (before | separator).
        Preserves order: first occurrence of each title is kept.

        Args:
            docs: List of documents in "title | content" format

        Returns:
            List of unique documents (by title)
        """
        seen_titles = set()
        unique_docs = []

        for doc in docs:
            # Extract title (everything before " | ")
            title = doc.split(" | ", 1)[0] if " | " in doc else doc

            # Normalize title for comparison
            normalized_title = title.strip().lower()

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

        return unique_docs


