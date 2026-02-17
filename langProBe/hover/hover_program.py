import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractorSignature(dspy.Signature):
    """Extract named entities from a claim and identify which entities have Wikipedia articles.
    Focus on people, places, organizations, creative works, and events that would have dedicated Wikipedia pages."""

    claim = dspy.InputField(desc="The factual claim to analyze")
    retrieved_titles = dspy.InputField(desc="List of Wikipedia article titles already retrieved (empty for hop 1)")

    claim_entities = dspy.OutputField(
        desc="List of named entities from the claim (people, places, organizations, works, events) "
             "that likely have Wikipedia articles. Include full names and specific identifiers."
    )
    retrieved_entities = dspy.OutputField(
        desc="List of entity names found in the retrieved Wikipedia article titles. "
             "Extract the main subject of each article title."
    )
    missing_entities = dspy.OutputField(
        desc="List of claim entities that do NOT have corresponding Wikipedia articles in retrieved_titles. "
             "These are entities we still need to find."
    )
    entity_relationships = dspy.OutputField(
        desc="List of relationships or connections mentioned in the claim between entities "
             "(e.g., 'X starred in Y', 'A is married to B', 'M was produced by N')"
    )


class EntityBridgeFinderSignature(dspy.Signature):
    """Identify potential bridge entities that could connect known entities to missing information.
    Bridge entities are intermediate entities that relate retrieved entities to missing entities."""

    claim = dspy.InputField(desc="The original claim being verified")
    entity_relationships = dspy.InputField(desc="Relationships between entities asserted in the claim")
    retrieved_entities = dspy.InputField(desc="Entities found in already-retrieved Wikipedia articles")
    missing_entities = dspy.InputField(desc="Entities from the claim that still need Wikipedia articles")
    hop_number = dspy.InputField(desc="Current hop number (2 or 3)")

    bridge_entities = dspy.OutputField(
        desc="List of potential bridge entities that could connect retrieved entities to missing entities. "
             "These might be: related people (family, colleagues), connecting works (movies, songs), "
             "organizations, events, or locations mentioned in relation to both known and missing entities."
    )
    bridge_reasoning = dspy.OutputField(
        desc="Explanation of how each bridge entity could help connect known facts to missing entities. "
             "For each bridge entity, explain the connection path."
    )


class BridgeQueryGeneratorSignature(dspy.Signature):
    """Generate a precise search query targeting specific bridge entities or missing entity combinations.
    Focus on entity names and their relationships rather than abstract topics."""

    claim = dspy.InputField(desc="The original claim being verified")
    missing_entities = dspy.InputField(desc="Entities that need Wikipedia article retrieval")
    bridge_entities = dspy.InputField(desc="Potential connecting entities identified")
    retrieved_titles = dspy.InputField(desc="Wikipedia titles already retrieved (to avoid)")
    hop_number = dspy.InputField(desc="Current hop number (2 or 3)")

    query = dspy.OutputField(
        desc="A search query targeting specific entity names. For hop 2, focus on bridge entities. "
             "For hop 3, focus on entity combinations (e.g., 'Entity A Entity B', 'X's Y'). "
             "Include full names and relationship keywords."
    )
    target_entity = dspy.OutputField(
        desc="The specific entity name (or entity combination) this query is designed to retrieve. "
             "This should match a Wikipedia article title."
    )
    rationale = dspy.OutputField(
        desc="Brief explanation of which missing entity this query targets and why this formulation should work."
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval system with bridge-entity discovery.
    Extracts entities from claims, tracks which entities have Wikipedia articles retrieved,
    identifies bridge entities that connect known facts to missing entities,
    and generates targeted queries for entity combinations.

    EVALUATION
    - Returns exactly 21 documents (7 per hop × 3 hops)
    - Tracks concrete entity names rather than abstract verification aspects
    - Explicitly targets missing Wikipedia articles for claim entities'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Entity-driven modules
        self.entity_extractor = dspy.ChainOfThought(EntityExtractorSignature)
        self.bridge_finder = dspy.ChainOfThought(EntityBridgeFinderSignature)
        self.bridge_query_generator = dspy.ChainOfThought(BridgeQueryGeneratorSignature)

    def _extract_titles(self, passages: list[str]) -> list[str]:
        """Extract document titles from passages in 'title | content' format"""
        return [passage.split(" | ")[0] for passage in passages]

    def forward(self, claim):
        # HOP 1: Direct claim retrieval + entity extraction
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = self._extract_titles(hop1_docs)

        # Extract entities from claim and hop 1 results
        entity_output_hop1 = self.entity_extractor(
            claim=claim,
            retrieved_titles=hop1_titles
        )

        # Track entity discovery
        claim_entities = entity_output_hop1.claim_entities
        retrieved_entities_hop1 = entity_output_hop1.retrieved_entities
        missing_entities_hop1 = entity_output_hop1.missing_entities
        entity_relationships = entity_output_hop1.entity_relationships
        all_retrieved_titles = hop1_titles.copy()

        # HOP 2: Bridge entity discovery
        bridge_output_hop2 = self.bridge_finder(
            claim=claim,
            entity_relationships=entity_relationships,
            retrieved_entities=retrieved_entities_hop1,
            missing_entities=missing_entities_hop1,
            hop_number="2"
        )

        # Generate bridge-targeted query
        hop2_query_output = self.bridge_query_generator(
            claim=claim,
            missing_entities=missing_entities_hop1,
            bridge_entities=bridge_output_hop2.bridge_entities,
            retrieved_titles=all_retrieved_titles,
            hop_number="2"
        )

        hop2_docs = self.retrieve_k(hop2_query_output.query).passages
        hop2_titles = self._extract_titles(hop2_docs)
        all_retrieved_titles.extend(hop2_titles)

        # Re-extract entities to update missing list
        entity_output_hop2 = self.entity_extractor(
            claim=claim,
            retrieved_titles=all_retrieved_titles
        )

        missing_entities_hop2 = entity_output_hop2.missing_entities
        retrieved_entities_hop2 = entity_output_hop2.retrieved_entities

        # HOP 3: Entity combination targeting
        # Find new bridges or combinations for remaining missing entities
        if missing_entities_hop2:
            bridge_output_hop3 = self.bridge_finder(
                claim=claim,
                entity_relationships=entity_relationships,
                retrieved_entities=retrieved_entities_hop2,
                missing_entities=missing_entities_hop2,
                hop_number="3"
            )
        else:
            # No missing entities, create empty prediction
            bridge_output_hop3 = dspy.Prediction(
                bridge_entities=[],
                bridge_reasoning="All claim entities have been retrieved"
            )

        hop3_query_output = self.bridge_query_generator(
            claim=claim,
            missing_entities=missing_entities_hop2,
            bridge_entities=bridge_output_hop3.bridge_entities,
            retrieved_titles=all_retrieved_titles,
            hop_number="3"
        )

        hop3_docs = self.retrieve_k(hop3_query_output.query).passages

        # Return all 21 documents (maintains evaluation contract)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
