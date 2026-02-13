import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_all_entities = dspy.Predict("claim->entities", desc="comma-separated list of all named entities, people, bands, works, and specific subjects mentioned or implied in the claim")
        self.create_query_hop2 = dspy.Predict("claim,entities,context->query")
        self.create_query_hop3 = dspy.Predict("claim,entities,context1,context2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # Extract all entities from the claim
        entities_str = self.extract_all_entities(claim=claim).entities
        entities_list = [e.strip() for e in entities_str.split(',')]

        # Split entities into 3 groups using round-robin
        entity_group1 = [entities_list[i] for i in range(0, len(entities_list), 3)]
        entity_group2 = [entities_list[i] for i in range(1, len(entities_list), 3)]
        entity_group3 = [entities_list[i] for i in range(2, len(entities_list), 3)]

        # HOP 1 - Use first entity group as query
        hop1_query = ", ".join(entity_group1) if entity_group1 else claim
        hop1_docs = self.retrieve_k(hop1_query).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2 - Use second entity group with context
        entity_group2_str = ", ".join(entity_group2) if entity_group2 else ""
        hop2_query = self.create_query_hop2(
            claim=claim, entities=entity_group2_str, context=summary_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Use third entity group with both contexts
        entity_group3_str = ", ".join(entity_group3) if entity_group3 else ""
        hop3_query = self.create_query_hop3(
            claim=claim, entities=entity_group3_str, context1=summary_1, context2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


