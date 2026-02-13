import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("claim,key_entities_1,key_facts_1->query")
        self.create_query_hop3 = dspy.Predict("claim,key_entities_1,key_facts_1,key_entities_2,key_facts_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_facts = dspy.Predict("claim,passages->key_entities,key_facts")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        extraction_1 = self.extract_facts(claim=claim, passages=hop1_docs)
        key_entities_1 = extraction_1.key_entities
        key_facts_1 = extraction_1.key_facts

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, key_entities_1=key_entities_1, key_facts_1=key_facts_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        extraction_2 = self.extract_facts(claim=claim, passages=hop2_docs)
        key_entities_2 = extraction_2.key_entities
        key_facts_2 = extraction_2.key_facts

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim,
            key_entities_1=key_entities_1,
            key_facts_1=key_facts_1,
            key_entities_2=key_entities_2,
            key_facts_2=key_facts_2,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


