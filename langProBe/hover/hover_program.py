import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction module
        self.extract_entities = dspy.ChainOfThought("claim -> key_entities: list[str]")

        # Coverage analysis module
        self.analyze_coverage = dspy.Predict(
            "claim, key_entities, retrieved_passages -> covered_entities: list[str], missing_entities: list[str]"
        )

        # Query generation modules for hop2 and hop3 (targeting missing entities)
        self.create_query_hop2 = dspy.ChainOfThought(
            "claim, missing_entities, covered_entities -> query"
        )
        self.create_query_hop3 = dspy.ChainOfThought(
            "claim, missing_entities, covered_entities -> query"
        )

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Extract key entities from the claim (3-5 critical entities/topics)
        key_entities = self.extract_entities(claim=claim).key_entities

        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages

        # Analyze coverage after hop 1
        coverage_hop1 = self.analyze_coverage(
            claim=claim,
            key_entities=key_entities,
            retrieved_passages=hop1_docs
        )
        covered_entities_hop1 = coverage_hop1.covered_entities
        missing_entities_hop1 = coverage_hop1.missing_entities

        # HOP 2 - Target missing entities
        hop2_query = self.create_query_hop2(
            claim=claim,
            missing_entities=missing_entities_hop1,
            covered_entities=covered_entities_hop1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Analyze coverage after hop 2
        coverage_hop2 = self.analyze_coverage(
            claim=claim,
            key_entities=key_entities,
            retrieved_passages=hop1_docs + hop2_docs
        )
        covered_entities_hop2 = coverage_hop2.covered_entities
        missing_entities_hop2 = coverage_hop2.missing_entities

        # HOP 3 - Target remaining missing entities
        hop3_query = self.create_query_hop3(
            claim=claim,
            missing_entities=missing_entities_hop2,
            covered_entities=covered_entities_hop2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


