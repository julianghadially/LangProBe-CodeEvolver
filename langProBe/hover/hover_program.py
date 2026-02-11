import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Claim decomposition module: breaks complex claims into atomic sub-claims
        self.decompose_claim = dspy.ChainOfThought("claim -> sub_claims: list[str]")

        # Query generation modules: create targeted queries for unverified sub-claims
        self.create_query_hop2 = dspy.Predict("sub_claim, retrieved_so_far -> query")
        self.create_query_hop3 = dspy.Predict("sub_claim, retrieved_so_far -> query")

        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Retrieve using full claim as before
        hop1_docs = self.retrieve_k(claim).passages

        # Decompose claim into 2-3 atomic sub-claims for targeted verification
        # Example: "X wrote Y which was released in Z and X is from location W"
        # becomes ["X wrote Y", "Y was released in Z", "X is from location W"]
        decomposition = self.decompose_claim(claim=claim)
        sub_claims = decomposition.sub_claims

        # Ensure we have at least 2-3 sub-claims for subsequent hops
        if not isinstance(sub_claims, list):
            sub_claims = [sub_claims]

        # HOP 2: Target the first unverified sub-claim
        # Generate query checking which entities/facts from first sub-claim are missing
        retrieved_so_far = "\n".join(hop1_docs)
        hop2_sub_claim = sub_claims[0] if len(sub_claims) > 0 else claim
        hop2_query = self.create_query_hop2(
            sub_claim=hop2_sub_claim,
            retrieved_so_far=retrieved_so_far
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Target the second unverified sub-claim
        # Generate query checking which entities/facts from second sub-claim are missing
        retrieved_so_far = "\n".join(hop1_docs + hop2_docs)
        hop3_sub_claim = sub_claims[1] if len(sub_claims) > 1 else (
            sub_claims[0] if len(sub_claims) > 0 else claim
        )
        hop3_query = self.create_query_hop3(
            sub_claim=hop3_sub_claim,
            retrieved_so_far=retrieved_so_far
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


