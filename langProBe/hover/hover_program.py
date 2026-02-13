import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_entities = dspy.Predict("claim->entities")
        self.create_query_hop2 = dspy.Predict("claim,entities,summary_1->query,target_entity")
        self.create_query_hop3 = dspy.Predict("claim,entities,summary_1,summary_2->query,target_entity")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # Extract entities from claim
        entities_result = self.extract_entities(claim=claim)
        entities = entities_result.entities

        # HOP 1 - Use first entity as query
        hop1_query = entities.split(',')[0].strip() if ',' in entities else entities.split()[0] if entities else claim
        hop1_docs = self.retrieve_k(hop1_query).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2 - Target a different entity
        hop2_result = self.create_query_hop2(claim=claim, entities=entities, summary_1=summary_1)
        hop2_query = hop2_result.query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Target remaining uncovered entities
        hop3_result = self.create_query_hop3(
            claim=claim, entities=entities, summary_1=summary_1, summary_2=summary_2
        )
        hop3_query = hop3_result.query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


