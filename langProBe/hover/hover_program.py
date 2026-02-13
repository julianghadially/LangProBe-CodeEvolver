import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtraction(dspy.Signature):
    """Extract 3-4 distinct named entities from the claim that require separate Wikipedia searches.
    Focus on specific people, places, events, and organizations that are central to verifying the claim.
    Each entity should be something that would have its own Wikipedia article."""

    claim = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="3-4 specific named entities (people, places, events, organizations) that need individual Wikipedia searches")
    reasoning = dspy.OutputField(desc="Brief explanation of why these entities are critical for verification")


class EntityTargeting(dspy.Signature):
    """Determine which specific entity to target next based on what information is still missing.
    Choose the entity most likely to fill gaps in verification evidence."""

    claim = dspy.InputField()
    entities = dspy.InputField(desc="List of entities extracted from the claim")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved")
    hop_number = dspy.InputField(desc="Current hop number (1, 2, or 3)")

    target_entity = dspy.OutputField(desc="The specific entity to search for in this hop")
    reasoning = dspy.OutputField(desc="Why this entity is the priority for this hop")
    search_query = dspy.OutputField(desc="Focused search query for this specific entity (not a broad claim reformulation)")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # 7 docs per hop = 21 total docs

        # Entity extraction with reasoning
        self.extract_entities = dspy.ChainOfThought(EntityExtraction)

        # Entity-targeted hop planning with reasoning
        self.target_entity = dspy.ChainOfThought(EntityTargeting)

        # Retriever
        self.retrieve_k = dspy.Retrieve(k=self.k)

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
        # Step 1: Extract 3-4 distinct named entities from the claim
        entity_result = self.extract_entities(claim=claim)
        entities = entity_result.entities

        all_docs = []
        retrieved_titles_set = set()

        # Step 2: Execute 3 entity-focused hops
        for hop_num in [1, 2, 3]:
            # Determine which entity to target in this hop
            retrieved_titles_str = ", ".join(sorted(retrieved_titles_set)) if retrieved_titles_set else "None yet"

            target_result = self.target_entity(
                claim=claim,
                entities=", ".join(entities) if isinstance(entities, list) else str(entities),
                retrieved_titles=retrieved_titles_str,
                hop_number=str(hop_num)
            )

            # Retrieve documents for the targeted entity with focused query
            query = target_result.search_query
            hop_docs = self.retrieve_k(query).passages

            # Track retrieved titles for next hop
            for doc in hop_docs:
                retrieved_titles_set.add(self._extract_title(doc))

            # Add to document collection
            all_docs.extend(hop_docs)

        # Step 3: Deduplicate by document title while preserving order
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


