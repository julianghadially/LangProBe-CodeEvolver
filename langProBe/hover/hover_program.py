import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimEntityTrackerSignature(dspy.Signature):
    """Extract key entities, relationships, and facts from a claim that need verification.
    Identify named entities (people, places, organizations, dates) and the relationships between them that the claim asserts."""

    claim = dspy.InputField(desc="The factual claim to analyze")
    entities = dspy.OutputField(desc="List of key entities mentioned or implied in the claim (people, places, organizations, dates, events)")
    key_facts = dspy.OutputField(desc="List of specific facts or relationships that need to be verified")
    verification_aspects = dspy.OutputField(desc="List of distinct aspects that need coverage (e.g., 'entity A existence', 'entity B properties', 'relationship between A and B')")


class CoverageAnalyzerSignature(dspy.Signature):
    """Analyze coverage of verification aspects based on retrieved documents.
    Identify which aspects are well-covered, which need more evidence, and which are missing."""

    claim = dspy.InputField(desc="The original claim being verified")
    verification_aspects = dspy.InputField(desc="List of aspects that need to be verified")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (to avoid duplicates)")
    passages = dspy.InputField(desc="The document passages retrieved in this hop")

    covered_aspects = dspy.OutputField(desc="Aspects that are well-covered by the retrieved documents")
    under_covered_aspects = dspy.OutputField(desc="Aspects mentioned but needing more evidence")
    missing_aspects = dspy.OutputField(desc="Aspects not yet addressed by retrieved documents")
    coverage_summary = dspy.OutputField(desc="Brief summary of what evidence has been found and what gaps remain")


class TargetedQueryGeneratorSignature(dspy.Signature):
    """Generate a targeted search query to fill coverage gaps.
    Focus on under-covered or missing aspects. Use negative signals from already-retrieved document titles to diversify results."""

    claim = dspy.InputField(desc="The original claim being verified")
    missing_aspects = dspy.InputField(desc="Verification aspects that have not been covered yet")
    under_covered_aspects = dspy.InputField(desc="Aspects needing more evidence")
    coverage_summary = dspy.InputField(desc="Summary of evidence found and gaps remaining")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (avoid retrieving again)")

    query = dspy.OutputField(desc="A search query focused on missing/under-covered aspects, formulated to retrieve documents different from those already obtained")
    rationale = dspy.OutputField(desc="Brief explanation of which gap this query targets")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval system with coverage-driven query expansion.
    Extracts entities from claims, tracks coverage across hops, and generates
    targeted queries to ensure diverse document retrieval.

    EVALUATION
    - Returns exactly 21 documents (7 per hop × 3 hops)
    - Preserves entity-level precision throughout retrieval
    - Explicitly avoids duplicate documents via title tracking'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Coverage-driven modules
        self.entity_tracker = dspy.ChainOfThought(ClaimEntityTrackerSignature)
        self.coverage_analyzer1 = dspy.ChainOfThought(CoverageAnalyzerSignature)
        self.coverage_analyzer2 = dspy.ChainOfThought(CoverageAnalyzerSignature)
        self.query_generator_hop2 = dspy.ChainOfThought(TargetedQueryGeneratorSignature)
        self.query_generator_hop3 = dspy.ChainOfThought(TargetedQueryGeneratorSignature)

    def _extract_titles(self, passages: list[str]) -> list[str]:
        """Extract document titles from passages in 'title | content' format"""
        return [passage.split(" | ")[0] for passage in passages]

    def forward(self, claim):
        # INITIALIZATION: Extract verification aspects
        tracker_output = self.entity_tracker(claim=claim)
        verification_aspects = tracker_output.verification_aspects
        all_retrieved_titles = []

        # HOP 1: Direct claim-based retrieval
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = self._extract_titles(hop1_docs)
        all_retrieved_titles.extend(hop1_titles)

        # ANALYZE HOP 1 COVERAGE
        coverage1 = self.coverage_analyzer1(
            claim=claim,
            verification_aspects=verification_aspects,
            retrieved_titles=hop1_titles,
            passages=hop1_docs
        )

        # HOP 2: Coverage-driven query
        hop2_query_output = self.query_generator_hop2(
            claim=claim,
            missing_aspects=coverage1.missing_aspects,
            under_covered_aspects=coverage1.under_covered_aspects,
            coverage_summary=coverage1.coverage_summary,
            retrieved_titles=all_retrieved_titles
        )

        hop2_docs = self.retrieve_k(hop2_query_output.query).passages
        hop2_titles = self._extract_titles(hop2_docs)
        all_retrieved_titles.extend(hop2_titles)

        # ANALYZE HOP 2 COVERAGE
        coverage2 = self.coverage_analyzer2(
            claim=claim,
            verification_aspects=verification_aspects,
            retrieved_titles=hop2_titles,
            passages=hop2_docs
        )

        # HOP 3: Final gap-filling query
        hop3_query_output = self.query_generator_hop3(
            claim=claim,
            missing_aspects=coverage2.missing_aspects,
            under_covered_aspects=coverage2.under_covered_aspects,
            coverage_summary=coverage2.coverage_summary,
            retrieved_titles=all_retrieved_titles
        )

        hop3_docs = self.retrieve_k(hop3_query_output.query).passages

        # Return all documents (maintains 21-doc contract)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
