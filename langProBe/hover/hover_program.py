import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.identify_missing_entities = dspy.Predict("claim,retrieved_docs->missing_entities")
        self.create_query_from_missing = dspy.Predict("claim,missing_entities->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Initial retrieval using the claim
        hop1_docs = self.retrieve_k(claim).passages

        # Identify missing entities after hop 1
        missing_entities_1 = self.identify_missing_entities(
            claim=claim, retrieved_docs=hop1_docs
        ).missing_entities

        # HOP 2: Generate query targeting missing entities
        hop2_query = self.create_query_from_missing(
            claim=claim, missing_entities=missing_entities_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Identify remaining missing entities after hop 1 + hop 2
        all_docs_so_far = hop1_docs + hop2_docs
        missing_entities_2 = self.identify_missing_entities(
            claim=claim, retrieved_docs=all_docs_so_far
        ).missing_entities

        # HOP 3: Generate query targeting remaining missing entities
        hop3_query = self.create_query_from_missing(
            claim=claim, missing_entities=missing_entities_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


