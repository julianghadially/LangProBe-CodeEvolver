import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractKeyEntities(dspy.Signature):
    """Extract key entities (names, titles, facts) as exact strings from passages.

    Focus on:
    - Person names (full names, titles)
    - Organization names
    - Location names
    - Dates and specific facts
    - Technical terms or concepts

    Extract entities that appear relevant to verifying the claim."""

    claim = dspy.InputField(desc="The claim to verify")
    passages = dspy.InputField(desc="Retrieved passages containing information")
    entities = dspy.OutputField(desc="List of key entities extracted as exact strings from passages")


class GenerateBridgingQuery(dspy.Signature):
    """Generate a search query to find documents that bridge/connect the found entities to answer the claim.

    The query should:
    - Connect the entities found so far
    - Look for relationships between entities
    - Target information that links entities to the claim"""

    claim = dspy.InputField(desc="The claim to verify")
    found_entities = dspy.InputField(desc="List of entities found so far")
    query = dspy.OutputField(desc="Search query to find bridging documents")


class GenerateMissingQuery(dspy.Signature):
    """Generate a search query targeting any missing information needed to verify the claim.

    The query should:
    - Identify gaps in the collected information
    - Target missing facts or connections
    - Look for information not yet covered by found entities"""

    claim = dspy.InputField(desc="The claim to verify")
    found_entities = dspy.InputField(desc="List of all entities found in previous hops")
    query = dspy.OutputField(desc="Search query to find missing information")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.retrieve_hop1 = dspy.Retrieve(k=10)
        self.retrieve_hop2 = dspy.Retrieve(k=10)
        self.retrieve_hop3 = dspy.Retrieve(k=10)
        self.extract_entities = dspy.ChainOfThought(ExtractKeyEntities)
        self.generate_bridging_query = dspy.ChainOfThought(GenerateBridgingQuery)
        self.generate_missing_query = dspy.ChainOfThought(GenerateMissingQuery)

    def forward(self, claim):
        # HOP 1: Initial retrieval and entity extraction
        hop1_docs = self.retrieve_hop1(claim).passages
        hop1_entities_result = self.extract_entities(
            claim=claim,
            passages=hop1_docs
        )
        hop1_entities = hop1_entities_result.entities

        # HOP 2: Bridging query based on hop 1 entities
        hop2_query_result = self.generate_bridging_query(
            claim=claim,
            found_entities=hop1_entities
        )
        hop2_query = hop2_query_result.query
        hop2_docs = self.retrieve_hop2(hop2_query).passages

        # Extract entities from hop 2
        hop2_entities_result = self.extract_entities(
            claim=claim,
            passages=hop2_docs
        )
        hop2_entities = hop2_entities_result.entities

        # Combine entities from hops 1 and 2
        combined_entities = hop1_entities + hop2_entities

        # HOP 3: Missing information query based on all entities so far
        hop3_query_result = self.generate_missing_query(
            claim=claim,
            found_entities=combined_entities
        )
        hop3_query = hop3_query_result.query
        hop3_docs = self.retrieve_hop3(hop3_query).passages

        # Combine all documents (30 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate by exact document text match, keeping first occurrence
        seen_texts = set()
        unique_docs = []
        for doc in all_docs:
            if doc not in seen_texts:
                seen_texts.add(doc)
                unique_docs.append(doc)

        # Keep first 21 unique documents
        final_docs = unique_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
