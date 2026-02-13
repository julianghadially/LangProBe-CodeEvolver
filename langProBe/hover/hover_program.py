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


class DocumentReranker(dspy.Signature):
    """Score a candidate document's relevance to verifying a claim based on the reasoning chain and missing information."""

    claim = dspy.InputField(desc="The claim that needs to be verified")
    reasoning_chain = dspy.InputField(desc="The accumulated reasoning about what information is needed")
    missing_info = dspy.InputField(desc="Specific information gaps identified during retrieval")
    candidate_doc = dspy.InputField(desc="The document to score for relevance")

    relevance_score = dspy.OutputField(desc="A relevance score from 0-10 indicating how useful this document is for verifying the claim")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 15
        self.query_planner_hop1 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop2 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.query_planner_hop3 = dspy.ChainOfThought(ChainOfThoughtQueryPlanner)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.reranker = dspy.ChainOfThought(DocumentReranker)

    def forward(self, claim):
        # Helper function to extract document title (text before " | ")
        def get_doc_title(doc):
            return doc.split(" | ")[0] if " | " in doc else doc

        # Helper function to deduplicate documents by title
        def deduplicate_docs(docs):
            seen_titles = set()
            unique_docs = []
            for doc in docs:
                title = get_doc_title(doc)
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_docs.append(doc)
            return unique_docs

        # HOP 1: Initial analysis and query generation
        hop1_plan = self.query_planner_hop1(
            claim=claim,
            retrieved_context=""
        )
        hop1_query = hop1_plan.next_query
        hop1_docs_raw = self.retrieve_k(hop1_query).passages
        hop1_docs = deduplicate_docs(hop1_docs_raw)
        hop1_context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])

        # HOP 2: Reason about what was found and what's missing
        hop2_plan = self.query_planner_hop2(
            claim=claim,
            retrieved_context=hop1_context
        )
        hop2_query = hop2_plan.next_query
        hop2_docs_raw = self.retrieve_k(hop2_query).passages
        # Deduplicate hop2 docs and remove any that overlap with hop1
        all_docs_so_far = hop1_docs.copy()
        seen_titles = {get_doc_title(doc) for doc in all_docs_so_far}
        hop2_docs = []
        for doc in hop2_docs_raw:
            title = get_doc_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                hop2_docs.append(doc)
                all_docs_so_far.append(doc)
        hop2_context = hop1_context + "\n\n" + "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop2_docs)])

        # HOP 3: Final targeted retrieval for remaining gaps
        hop3_plan = self.query_planner_hop3(
            claim=claim,
            retrieved_context=hop2_context
        )
        hop3_query = hop3_plan.next_query
        hop3_docs_raw = self.retrieve_k(hop3_query).passages
        # Deduplicate hop3 docs and remove any that overlap with hop1 or hop2
        hop3_docs = []
        for doc in hop3_docs_raw:
            title = get_doc_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                hop3_docs.append(doc)
                all_docs_so_far.append(doc)

        # Combine all unique documents
        all_unique_docs = hop1_docs + hop2_docs + hop3_docs

        # Build reasoning chain context from all hops
        reasoning_chain = hop1_plan.reasoning
        missing_info = f"{hop1_plan.missing_information}\n{hop2_plan.missing_information}\n{hop3_plan.missing_information}"

        # Rerank all documents
        scored_docs = []
        for doc in all_unique_docs:
            rerank_result = self.reranker(
                claim=claim,
                reasoning_chain=reasoning_chain,
                missing_info=missing_info,
                candidate_doc=doc
            )
            # Extract numeric score from relevance_score string
            try:
                score = float(rerank_result.relevance_score.split()[0])
            except (ValueError, IndexError, AttributeError):
                # If we can't parse a numeric score, try to extract first number
                import re
                match = re.search(r'\d+(?:\.\d+)?', str(rerank_result.relevance_score))
                score = float(match.group()) if match else 0.0
            scored_docs.append((score, doc))

        # Sort by relevance score (descending) and take top 21
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_21_docs = [doc for score, doc in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)


