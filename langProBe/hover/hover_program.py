import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractor(dspy.Signature):
    """Extract 2-3 key named entities or concepts from the claim that need verification.

    Focus on identifying the main entities (people, places, organizations, concepts) that are
    central to verifying the claim. These should be concrete entities that can be searched for.
    """

    claim = dspy.InputField(desc="The claim that needs to be verified")
    entities = dspy.OutputField(desc="A list of 2-3 key named entities/concepts from the claim (comma-separated)")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused Wikipedia search query for a specific entity.

    Create a targeted query that will retrieve relevant information about the given entity
    in the context of the original claim.
    """

    claim = dspy.InputField(desc="The original claim being verified")
    entity = dspy.InputField(desc="The specific entity to search for")
    query = dspy.OutputField(desc="A focused Wikipedia search query for this entity")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # 7 docs per search, 3 searches = 21 docs max
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.query_generator = dspy.Predict(EntityQueryGenerator)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Step 1: Extract key entities from the claim
        extraction_result = self.entity_extractor(claim=claim)
        entities_str = extraction_result.entities

        # Parse entities (comma-separated) and take up to 3
        entities = [e.strip() for e in entities_str.split(',')][:3]

        # Step 2: Track seen documents for deduplication
        seen_docs = set()
        all_docs = []

        # Step 3: Generate focused query for each entity and retrieve
        for entity in entities:
            # Generate entity-specific query
            query_result = self.query_generator(claim=claim, entity=entity)
            query = query_result.query

            # Retrieve documents for this entity
            retrieved_passages = self.retrieve_k(query).passages

            # Deduplicate by content
            for doc in retrieved_passages:
                # Use the document content as the deduplication key
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    all_docs.append(doc)

        # Step 4: Truncate to exactly 21 documents
        deduplicated_docs = all_docs[:21]

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


