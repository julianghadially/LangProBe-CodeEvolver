import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class DiverseQueryGenerator(dspy.Signature):
    """Generate diverse search queries targeting different aspects of the claim to improve document retrieval coverage.

    Generate queries that target:
    1. Entities and people mentioned in the claim
    2. Locations and places related to the claim
    3. Events, works, or connections mentioned in the claim

    Use the previously retrieved document titles to avoid redundancy and explore new information spaces."""

    claim: str = dspy.InputField(desc="the factual claim to verify")
    retrieved_titles: str = dspy.InputField(desc="titles of documents already retrieved, to avoid redundancy")
    query_entity: str = dspy.OutputField(desc="query targeting entities and people")
    query_location: str = dspy.OutputField(desc="query targeting locations and places")
    query_connection: str = dspy.OutputField(desc="query targeting events, works, or connections")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.retrieve_hop1 = dspy.Retrieve(k=7)
        self.retrieve_hop2 = dspy.Retrieve(k=4)
        self.retrieve_hop3 = dspy.Retrieve(k=3)
        self.diverse_query_gen = dspy.Predict(DiverseQueryGenerator)

    def forward(self, claim):
        # HOP 1: Direct retrieval from claim (k=7)
        hop1_docs = self.retrieve_hop1(claim).passages
        hop1_titles = " | ".join([doc.split(" | ")[0] for doc in hop1_docs])

        # HOP 2: Generate 2 diverse queries, retrieve k=4 per query (total 8 docs)
        hop2_diverse = self.diverse_query_gen(claim=claim, retrieved_titles=hop1_titles)
        hop2_docs_entity = self.retrieve_hop2(hop2_diverse.query_entity).passages
        hop2_docs_location = self.retrieve_hop2(hop2_diverse.query_location).passages
        hop2_docs = hop2_docs_entity + hop2_docs_location

        hop2_titles = " | ".join([doc.split(" | ")[0] for doc in hop2_docs])
        combined_titles_hop1_hop2 = hop1_titles + " | " + hop2_titles

        # HOP 3: Generate 2 diverse queries, retrieve k=3 per query (total 6 docs)
        hop3_diverse = self.diverse_query_gen(claim=claim, retrieved_titles=combined_titles_hop1_hop2)
        hop3_docs_entity = self.retrieve_hop3(hop3_diverse.query_entity).passages
        hop3_docs_connection = self.retrieve_hop3(hop3_diverse.query_connection).passages
        hop3_docs = hop3_docs_entity + hop3_docs_connection

        # Total: 7 + 8 + 6 = 21 documents
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
