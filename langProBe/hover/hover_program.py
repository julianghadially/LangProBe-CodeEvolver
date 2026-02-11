import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimComplexitySignature(dspy.Signature):
    """Analyze the claim to estimate its complexity for multi-hop fact verification.

    Consider:
    - Number of distinct entities/topics that need to be verified (people, places, organizations, events)
    - Whether the claim involves relationships between multiple entities
    - Temporal or causal relationships that require evidence from different sources

    Return num_entities on a 1-5 scale where:
    1-2: Simple claim about a single entity or straightforward fact
    3: Moderate claim involving 2-3 related entities or concepts
    4-5: Complex claim with multiple entities, relationships, or multi-faceted verification needs

    Return complexity_score on a 1-5 scale reflecting overall verification difficulty.
    """

    claim = dspy.InputField(desc="The claim to analyze")
    num_entities: int = dspy.OutputField(desc="Number of distinct entities/topics needing verification (1-5 scale)")
    complexity_score: int = dspy.OutputField(desc="Overall complexity score for verification (1-5 scale)")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        # Complexity analyzer
        self.analyze_complexity = dspy.ChainOfThought(ClaimComplexitySignature)

        # Query generation modules
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")

        # Summarization modules
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def _allocate_k_values(self, num_entities: int) -> tuple[int, int, int]:
        """Dynamically allocate k values across 3 hops based on claim complexity.

        Strategy:
        - Simple claims (1-2 entities): k=[10, 8, 3] - Go deep early on the main entity
        - Moderate claims (3 entities): k=[7, 7, 7] - Balanced coverage across hops
        - Complex claims (4+ entities): k=[5, 8, 8] - Cast wider net across multiple hops

        Args:
            num_entities: Number of entities/topics to verify (1-5 scale)

        Returns:
            Tuple of (k1, k2, k3) values that sum to 21
        """
        # Ensure num_entities is in valid range
        num_entities = max(1, min(5, num_entities))

        if num_entities <= 2:
            # Simple claims: go deep early
            return (10, 8, 3)
        elif num_entities == 3:
            # Moderate claims: balanced coverage
            return (7, 7, 7)
        else:  # num_entities >= 4
            # Complex claims: wider net across hops
            return (5, 8, 8)

    def forward(self, claim):
        # Analyze claim complexity to determine k-value allocation
        complexity_analysis = self.analyze_complexity(claim=claim)
        num_entities = complexity_analysis.num_entities
        complexity_score = complexity_analysis.complexity_score

        # Allocate k values dynamically based on complexity
        k1, k2, k3 = self._allocate_k_values(num_entities)

        # HOP 1: Retrieve with dynamic k1
        retrieve_hop1 = dspy.Retrieve(k=k1)
        hop1_docs = retrieve_hop1(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Retrieve with dynamic k2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        retrieve_hop2 = dspy.Retrieve(k=k2)
        hop2_docs = retrieve_hop2(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Retrieve with dynamic k3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        retrieve_hop3 = dspy.Retrieve(k=k3)
        hop3_docs = retrieve_hop3(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


