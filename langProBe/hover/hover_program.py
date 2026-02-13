import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ChainOfThoughtQueryPlanner(dspy.Signature):
    """Analyze the claim and retrieved context to strategically plan the next retrieval query.

    Decompose what entities, relationships, and facts are needed to verify the claim.
    Analyze what information has already been found versus what's still missing.
    Generate a targeted query to find the specific missing information needed for the next hop.
    """

    claim = dspy.InputField(desc="The claim that needs to be verified through multi-hop reasoning")
    retrieved_context = dspy.InputField(desc="The context retrieved so far from previous hops (may be empty for first hop)")

    reasoning = dspy.OutputField(desc="Explain the multi-hop reasoning chain needed: what entities/relationships are mentioned in the claim and how they connect")
    missing_information = dspy.OutputField(desc="Identify specific gaps: what key information was found in retrieved_context vs. what's still needed to verify the claim")
    next_query = dspy.OutputField(desc="A focused search query to find the specific missing information identified above")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Initial analysis and query generation
        hop1_plan = self.query_planner_hop1(
            claim=claim,
            retrieved_context=""
        )
        hop1_query = hop1_plan.next_query
        hop1_docs = self.retrieve_k(hop1_query).passages
        hop1_context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])

        # HOP 2: Reason about what was found and what's missing
        hop2_plan = self.query_planner_hop2(
            claim=claim,
            retrieved_context=hop1_context
        )
        hop2_query = hop2_plan.next_query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop2_docs)])

        # HOP 3: Final targeted retrieval for remaining gaps
        hop3_plan = self.query_planner_hop3(
            claim=claim,
            retrieved_context=hop2_context
        )
        hop3_query = hop3_plan.next_query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


