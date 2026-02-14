import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtraction(dspy.Signature):
    """Extract 3-4 distinct named entities from the claim that require separate Wikipedia searches.
    Focus on specific people, places, events, and organizations that are central to verifying the claim.
    Each entity should be something that would have its own Wikipedia article."""

    claim = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="3-4 specific named entities (people, places, events, organizations) that need individual Wikipedia searches")
    reasoning = dspy.OutputField(desc="Brief explanation of why these entities are critical for verification")


class ContextualBridgeQuery(dspy.Signature):
    """Generate 1-2 contextual bridge queries to find connections between entities and fill verification gaps.
    These queries should explore relationships, events, or context that link the entities together."""

    claim = dspy.InputField()
    entities = dspy.InputField(desc="List of entities extracted from the claim")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved from direct entity searches")

    bridge_queries: list[str] = dspy.OutputField(desc="1-2 contextual queries to find missing connections between entities")
    reasoning = dspy.OutputField(desc="Explanation of what connections or context these queries aim to find")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()

        # Stage 1: Direct entity retrieval (k=3 docs per entity, 3-4 entities = 9-12 docs)
        self.entity_k = 3

        # Stage 2: Contextual bridging (k=5-6 docs per bridge query, 1-2 queries = 5-12 docs)
        self.bridge_k = 5  # Will use 5-6 depending on number of bridge queries

        # Entity extraction with reasoning
        self.extract_entities = dspy.ChainOfThought(EntityExtraction)

        # Contextual bridge query generation with reasoning
        self.generate_bridge_queries = dspy.ChainOfThought(ContextualBridgeQuery)

        # Retrievers for each stage
        self.retrieve_entity = dspy.Retrieve(k=self.entity_k)
        self.retrieve_bridge = dspy.Retrieve(k=self.bridge_k)

    def _extract_title(self, doc):
        """Extract the title (first line) from a document."""
        if "\n" in doc:
            return doc.split("\n")[0]
        return doc[:100]

    def _deduplicate_by_title(self, docs):
        """Deduplicate documents by title while preserving order."""
        seen_titles = set()
        deduplicated = []

        for doc in docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(doc)

        return deduplicated

    def forward(self, claim):
        # STAGE 1: Direct Entity Retrieval
        # Extract 3-4 named entities from the claim
        entity_result = self.extract_entities(claim=claim)
        entities = entity_result.entities

        all_docs = []
        retrieved_titles_set = set()

        # Retrieve k=3 docs per entity using exact entity names as queries
        for entity in entities:
            # Use exact entity name as query (not a contextual query)
            entity_docs = self.retrieve_entity(entity).passages
            all_docs.extend(entity_docs)

            # Track retrieved titles
            for doc in entity_docs:
                retrieved_titles_set.add(self._extract_title(doc))

        # STAGE 2: Contextual Bridging
        # Generate 1-2 contextual bridge queries to find missing connections
        retrieved_titles_str = ", ".join(sorted(retrieved_titles_set))
        entities_str = ", ".join(entities) if isinstance(entities, list) else str(entities)

        bridge_result = self.generate_bridge_queries(
            claim=claim,
            entities=entities_str,
            retrieved_titles=retrieved_titles_str
        )
        bridge_queries = bridge_result.bridge_queries

        # Adjust k for bridge queries to reach 19-21 total docs
        # If 1 bridge query: use k=6, if 2 bridge queries: use k=5-6
        num_bridge_queries = len(bridge_queries)
        if num_bridge_queries == 1:
            bridge_k = 6
        else:  # 2 queries
            bridge_k = 5

        # Retrieve k=5-6 docs per bridge query
        for bridge_query in bridge_queries:
            bridge_docs = dspy.Retrieve(k=bridge_k)(bridge_query).passages
            all_docs.extend(bridge_docs)

            # Track retrieved titles
            for doc in bridge_docs:
                retrieved_titles_set.add(self._extract_title(doc))

        # Deduplicate by title at the end
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


