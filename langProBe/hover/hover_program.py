import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate


class EntityExtractor(dspy.Signature):
    """Extract key named entities, titles, and specific terms from retrieved passages
    that are relevant to the claim. Focus on proper nouns, specific names, dates,
    locations, organizations, and technical terms that should be preserved for
    subsequent searches."""

    claim = dspy.InputField(desc="The original claim to verify")
    passages = dspy.InputField(desc="Retrieved passages from previous hop")
    entities: list[str] = dspy.OutputField(
        desc="List of key named entities, titles, and specific terms extracted from passages"
    )


class CoverageQueryGenerator(dspy.Signature):
    """Generate a search query that identifies information gaps and seeks missing
    evidence. The query should aim to find complementary information not yet
    covered by the retrieved entities."""

    claim = dspy.InputField(desc="The original claim to verify")
    retrieved_entities = dspy.InputField(desc="Entities and information already retrieved")
    query: str = dspy.OutputField(
        desc="Search query to find missing information and fill coverage gaps"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()

        # Hop 1: retrieve with original claim
        self.retrieve_hop1 = dspy.Retrieve(k=10)

        # Hop 2: dual-query strategy
        self.extract_entities_hop2 = dspy.ChainOfThought(EntityExtractor)
        self.reformulate_query_hop2 = dspy.ChainOfThought("claim->query")
        self.retrieve_hop2 = dspy.Retrieve(k=8)

        # Hop 3: dual-query strategy
        self.extract_entities_hop3 = dspy.ChainOfThought(EntityExtractor)
        self.generate_coverage_query_hop3 = dspy.ChainOfThought(CoverageQueryGenerator)
        self.retrieve_hop3 = dspy.Retrieve(k=8)

    def forward(self, claim):
        # HOP 1: Retrieve k=10 documents using the original claim
        hop1_docs = self.retrieve_hop1(claim).passages

        # HOP 2: Dual-query strategy
        # Query 1: Entity-focused query from hop 1 documents
        entities_hop1 = self.extract_entities_hop2(
            claim=claim,
            passages=hop1_docs
        ).entities

        # Create entity search query by joining extracted entities
        entity_query_hop2 = " ".join(entities_hop1) if entities_hop1 else claim
        hop2_entity_docs = self.retrieve_hop2(entity_query_hop2).passages

        # Query 2: Broader reformulated claim query
        reformulated_query_hop2 = self.reformulate_query_hop2(claim=claim).query
        hop2_reformulated_docs = self.retrieve_hop2(reformulated_query_hop2).passages

        # Merge and deduplicate to keep top 10 unique documents
        hop2_docs = deduplicate(hop2_entity_docs + hop2_reformulated_docs)[:10]

        # HOP 3: Dual-query strategy
        # Query 1: Entity-focused query from hop 2 documents
        entities_hop2 = self.extract_entities_hop3(
            claim=claim,
            passages=hop2_docs
        ).entities

        entity_query_hop3 = " ".join(entities_hop2) if entities_hop2 else claim
        hop3_entity_docs = self.retrieve_hop3(entity_query_hop3).passages

        # Query 2: Coverage-oriented query to identify information gaps
        all_retrieved_entities = entities_hop1 + entities_hop2 if entities_hop2 else entities_hop1
        retrieved_entities_str = ", ".join(all_retrieved_entities) if all_retrieved_entities else "None yet"

        coverage_query_hop3 = self.generate_coverage_query_hop3(
            claim=claim,
            retrieved_entities=retrieved_entities_str
        ).query
        hop3_coverage_docs = self.retrieve_hop3(coverage_query_hop3).passages

        # Merge and deduplicate to keep top 10 unique documents
        hop3_docs = deduplicate(hop3_entity_docs + hop3_coverage_docs)[:10]

        # Concatenate all unique documents from all hops (max 21: 10+10+10, but deduplicate across hops)
        all_docs = deduplicate(hop1_docs + hop2_docs + hop3_docs)[:21]

        return dspy.Prediction(retrieved_docs=all_docs)
