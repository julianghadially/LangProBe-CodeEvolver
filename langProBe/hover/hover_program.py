import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_key_info = dspy.Predict("claim, passages -> key_entities, key_titles")
        self.create_query_hop2 = dspy.Predict("claim, key_entities_1, key_titles_1 -> query")
        self.create_query_hop3 = dspy.Predict("claim, key_entities_1, key_titles_1, key_entities_2, key_titles_2 -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        hop1_info = self.extract_key_info(claim=claim, passages=hop1_docs)
        key_entities_1 = hop1_info.key_entities
        key_titles_1 = hop1_info.key_titles

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, key_entities_1=key_entities_1, key_titles_1=key_titles_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_info = self.extract_key_info(claim=claim, passages=hop2_docs)
        key_entities_2 = hop2_info.key_entities
        key_titles_2 = hop2_info.key_titles

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim,
            key_entities_1=key_entities_1,
            key_titles_1=key_titles_1,
            key_entities_2=key_entities_2,
            key_titles_2=key_titles_2,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


