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
        self.k_hop1 = 15  # More diverse initial candidates
        self.k_hop2_3 = 9  # Standard k for subsequent hops
        self.k_followup = 5  # For additional retrieval when deduplication leaves gaps
        self.target_unique_docs_per_hop = 7  # Target unique documents per hop after first
        self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.retrieve_hop1 = dspy.Retrieve(k=self.k_hop1)
        self.retrieve_hop2_3 = dspy.Retrieve(k=self.k_hop2_3)
        self.retrieve_followup = dspy.Retrieve(k=self.k_followup)

    def _extract_title(self, doc):
        """Extract document title by splitting on ' | ' and taking first part."""
        return doc.split(" | ")[0] if " | " in doc else doc

    def _deduplicate_and_retrieve_more(self, initial_docs, query, seen_titles, target_count, retrieve_fn):
        """Deduplicate documents and retrieve more if needed to reach target count.

        Args:
            initial_docs: List of initially retrieved documents
            query: The query to use for additional retrievals
            seen_titles: Set of already-seen document titles
            target_count: Target number of unique new documents
            retrieve_fn: Function to call for additional retrieval

        Returns:
            List of unique new documents (deduplicated)
        """
        unique_docs = []

        # First pass: deduplicate initial documents
        for doc in initial_docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

        # If we have fewer than target, try up to 2 additional retrievals
        retrieval_attempts = 0
        max_attempts = 2

        while len(unique_docs) < target_count and retrieval_attempts < max_attempts:
            retrieval_attempts += 1
            additional_docs = retrieve_fn(query).passages

            for doc in additional_docs:
                if len(unique_docs) >= target_count:
                    break
                title = self._extract_title(doc)
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)

        return unique_docs

    def forward(self, claim):
        # Track all seen document titles across hops
        seen_titles = set()

        # HOP 1: Initial analysis and query generation (k=15 for diversity)
        hop1_plan = self.query_planner_hop1(
            claim=claim,
            retrieved_context=""
        )
        hop1_query = hop1_plan.next_query
        hop1_docs_raw = self.retrieve_hop1(hop1_query).passages

        # For hop 1, just track titles (no deduplication since it's the first hop)
        hop1_docs = []
        for doc in hop1_docs_raw:
            title = self._extract_title(doc)
            seen_titles.add(title)
            hop1_docs.append(doc)

        hop1_context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])

        # HOP 2: Reason about what was found and what's missing (k=9, deduplicate, target 7 unique)
        hop2_plan = self.query_planner_hop2(
            claim=claim,
            retrieved_context=hop1_context
        )
        hop2_query = hop2_plan.next_query
        hop2_docs_raw = self.retrieve_hop2_3(hop2_query).passages

        # Deduplicate and retrieve more if needed
        hop2_docs = self._deduplicate_and_retrieve_more(
            hop2_docs_raw,
            hop2_query,
            seen_titles,
            self.target_unique_docs_per_hop,
            self.retrieve_followup
        )

        hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop2_docs)])

        # HOP 3: Final targeted retrieval for remaining gaps (k=9, deduplicate, target 7 unique)
        hop3_plan = self.query_planner_hop3(
            claim=claim,
            retrieved_context=hop2_context
        )
        hop3_query = hop3_plan.next_query
        hop3_docs_raw = self.retrieve_hop2_3(hop3_query).passages

        # Deduplicate and retrieve more if needed
        hop3_docs = self._deduplicate_and_retrieve_more(
            hop3_docs_raw,
            hop3_query,
            seen_titles,
            self.target_unique_docs_per_hop,
            self.retrieve_followup
        )

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


