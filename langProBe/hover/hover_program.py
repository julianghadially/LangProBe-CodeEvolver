import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractionHop1Signature(dspy.Signature):
    """Extract 3-5 key entities from retrieved passages that are mentioned in the claim
    but need more information. Focus on specific names, organizations, dates, locations,
    or titles that would benefit from additional retrieval."""

    claim = dspy.InputField(desc="The claim to verify")
    passages = dspy.InputField(desc="Retrieved passages from hop 1")
    entities: list[str] = dspy.OutputField(
        desc="List of 3-5 key entities (people, bands, organizations, dates, titles) that need more context"
    )


class EntityExtractionHop2Signature(dspy.Signature):
    """Extract bridging entities from hop 2 passages that connect to the original entities.
    These should be new entities that help establish relationships with entities from hop 1."""

    claim = dspy.InputField(desc="The claim to verify")
    original_entities = dspy.InputField(desc="Entities extracted from hop 1")
    passages = dspy.InputField(desc="Retrieved passages from hop 2")
    bridging_entities: list[str] = dspy.OutputField(
        desc="List of 3-5 new entities that bridge/connect to the original entities"
    )


class QueryHop2Signature(dspy.Signature):
    """Generate a search query that explores relationships between the claim and extracted entities."""

    claim = dspy.InputField(desc="The claim to verify")
    entities = dspy.InputField(desc="Key entities from hop 1 that need more context")
    query = dspy.OutputField(desc="Search query asking about relationships between these entities")


class QueryHop3Signature(dspy.Signature):
    """Generate a search query that explicitly connects original claim entities to bridging entities."""

    claim = dspy.InputField(desc="The claim to verify")
    original_entities = dspy.InputField(desc="Entities from hop 1")
    bridging_entities = dspy.InputField(desc="Bridging entities from hop 2")
    query = dspy.OutputField(
        desc="Search query asking about the connection between original and bridging entities"
    )


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction modules (use ChainOfThought for better reasoning)
        self.extract_entities_hop1 = dspy.ChainOfThought(EntityExtractionHop1Signature)
        self.extract_entities_hop2 = dspy.ChainOfThought(EntityExtractionHop2Signature)

        # Query generation modules (use ChainOfThought for complex reasoning)
        self.create_query_hop2 = dspy.ChainOfThought(QueryHop2Signature)
        self.create_query_hop3 = dspy.ChainOfThought(QueryHop3Signature)

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Initial retrieval with claim
        hop1_docs = self.retrieve_k(claim).passages

        # Extract key entities from hop1 that need more information
        entities_1_result = self.extract_entities_hop1(
            claim=claim,
            passages=hop1_docs
        )
        entities_1 = entities_1_result.entities  # list[str]

        # HOP 2: Query about relationships between claim and extracted entities
        hop2_query_result = self.create_query_hop2(
            claim=claim,
            entities=", ".join(entities_1)  # Convert list to comma-separated string
        )
        hop2_query = hop2_query_result.query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Extract bridging entities from hop2 that connect to original entities
        entities_2_result = self.extract_entities_hop2(
            claim=claim,
            original_entities=", ".join(entities_1),
            passages=hop2_docs
        )
        bridging_entities = entities_2_result.bridging_entities  # list[str]

        # HOP 3: Query about connection between original and bridging entities
        hop3_query_result = self.create_query_hop3(
            claim=claim,
            original_entities=", ".join(entities_1),
            bridging_entities=", ".join(bridging_entities)
        )
        hop3_query = hop3_query_result.query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Return all retrieved documents (same output structure as before)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


