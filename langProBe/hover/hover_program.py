import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class InitialQueryGeneration(dspy.Signature):
    """Analyze the claim to generate a comprehensive search query covering all aspects needed for verification.
    The query should capture all key concepts, entities, events, and relationships mentioned in the claim."""

    claim = dspy.InputField()
    search_query = dspy.OutputField(desc="Comprehensive search query that covers all aspects of the claim")
    reasoning = dspy.OutputField(desc="Brief explanation of what aspects of the claim this query addresses")


class GapAnalysis(dspy.Signature):
    """Analyze what key facts are still missing from the retrieved documents to verify the claim.
    Compare the claim against all retrieved documents to identify gaps in coverage."""

    claim = dspy.InputField()
    all_retrieved_docs_so_far = dspy.InputField(desc="All documents retrieved in previous hops")

    missing_information = dspy.OutputField(desc="Specific facts, connections, or evidence still needed to verify the claim")
    next_search_query = dspy.OutputField(desc="Targeted search query to find the missing information")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()

        # Initial comprehensive query generation
        self.generate_initial_query = dspy.ChainOfThought(InitialQueryGeneration)

        # Gap analysis for subsequent hops
        self.analyze_gaps = dspy.ChainOfThought(GapAnalysis)

        # Retrievers with different k values per hop
        self.retrieve_k10 = dspy.Retrieve(k=10)  # For hops 1 and 2
        self.retrieve_k15 = dspy.Retrieve(k=15)  # For hop 3

    def _extract_title(self, doc):
        """Extract the title (first line) from a document."""
        if "\n" in doc:
            return doc.split("\n")[0]
        return doc[:100]

    def _deduplicate_by_title(self, docs):
        """Deduplicate documents by title while preserving order."""
        seen_titles = set()
        deduplicated = []

        for doc in docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(doc)

        return deduplicated

    def forward(self, claim):
        all_docs = []

        # Hop 1: Generate comprehensive initial query covering all aspects of the claim
        initial_query_result = self.generate_initial_query(claim=claim)
        hop1_query = initial_query_result.search_query

        # Retrieve k=10 documents for hop 1
        hop1_docs = self.retrieve_k10(hop1_query).passages
        all_docs.extend(hop1_docs)

        # Hop 2: Analyze gaps in retrieved documents and generate targeted query
        all_docs_text = "\n\n---\n\n".join(all_docs)
        gap_analysis_hop2 = self.analyze_gaps(
            claim=claim,
            all_retrieved_docs_so_far=all_docs_text
        )

        # Retrieve k=10 documents for hop 2 based on gap analysis
        hop2_query = gap_analysis_hop2.next_search_query
        hop2_docs = self.retrieve_k10(hop2_query).passages
        all_docs.extend(hop2_docs)

        # Hop 3: Perform another gap analysis with all documents retrieved so far
        all_docs_text = "\n\n---\n\n".join(all_docs)
        gap_analysis_hop3 = self.analyze_gaps(
            claim=claim,
            all_retrieved_docs_so_far=all_docs_text
        )

        # Retrieve k=15 documents for hop 3 to allow more coverage
        hop3_query = gap_analysis_hop3.next_search_query
        hop3_docs = self.retrieve_k15(hop3_query).passages
        all_docs.extend(hop3_docs)

        # Deduplicate by document title while preserving order (stays under 21 after dedup)
        deduplicated_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=deduplicated_docs)


