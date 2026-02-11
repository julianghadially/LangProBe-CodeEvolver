import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_entities = dspy.ChainOfThought("claim -> entities: list[str], reasoning: str")
        self.extract_entities.__doc__ = "Extract all named entities (people, places, organizations, works) mentioned directly or indirectly in the claim that would need Wikipedia articles for verification."
        self.select_docs_hop1 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.select_docs_hop2 = dspy.ChainOfThought("claim, passages -> selected_passages: list[str], reasoning: str")
        self.create_query_hop2 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, selected_passages -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # ENTITY EXTRACTION
        entities_result = self.extract_entities(claim=claim)
        entity_names = entities_result.entities

        # Retrieve Wikipedia articles for each entity
        entity_docs = []
        entity_retrieve = dspy.Retrieve(k=1)
        for entity_name in entity_names:
            entity_result = entity_retrieve(entity_name)
            entity_docs.extend(entity_result.passages)

        # Calculate dynamic k for remaining hops to maintain total of 21 docs
        dynamic_k = max(1, (21 - len(entity_docs)) // 3)
        retrieve_dynamic = dspy.Retrieve(k=dynamic_k)

        # HOP 1
        hop1_docs = retrieve_dynamic(claim).passages
        selection_1 = self.select_docs_hop1(
            claim=claim, passages=hop1_docs
        )
        selected_passages_1 = selection_1.selected_passages

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, selected_passages=selected_passages_1
        ).query
        hop2_docs = retrieve_dynamic(hop2_query).passages
        selection_2 = self.select_docs_hop2(
            claim=claim, passages=hop2_docs
        )
        selected_passages_2 = selection_2.selected_passages

        # HOP 3
        # Combine selected passages from hop1 and hop2 for context
        combined_selected = selected_passages_1 + selected_passages_2
        hop3_query = self.create_query_hop3(
            claim=claim, selected_passages=combined_selected
        ).query
        hop3_docs = retrieve_dynamic(hop3_query).passages

        # Deduplicate: remove hop docs that are already in entity_docs
        entity_docs_set = set(entity_docs)
        hop1_docs_dedup = [doc for doc in hop1_docs if doc not in entity_docs_set]
        hop2_docs_dedup = [doc for doc in hop2_docs if doc not in entity_docs_set]
        hop3_docs_dedup = [doc for doc in hop3_docs if doc not in entity_docs_set]

        return dspy.Prediction(retrieved_docs=entity_docs + hop1_docs_dedup + hop2_docs_dedup + hop3_docs_dedup)


