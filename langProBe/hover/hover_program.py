import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class CoverageAssessment(dspy.Signature):
    """Assess how well the retrieved passages cover all aspects needed to verify or understand the claim.
    Provide a confidence score (0-1), identify missing entities or topics, and summarize coverage gaps."""

    claim: str = dspy.InputField(desc="the claim to verify or understand")
    retrieved_passages: str = dspy.InputField(desc="the passages retrieved so far")
    confidence_score: float = dspy.OutputField(desc="confidence that all relevant information has been retrieved (0.0 to 1.0)")
    missing_entities: list[str] = dspy.OutputField(desc="list of entities, topics, or aspects not yet covered")
    coverage_summary: str = dspy.OutputField(desc="summary of what has been covered and what is still missing")


class AdaptiveQueryGenerator(dspy.Signature):
    """Generate a focused search query to retrieve information about missing aspects of the claim.
    Target specific entities or topics that have not been adequately covered."""

    claim: str = dspy.InputField(desc="the original claim to verify or understand")
    coverage_summary: str = dspy.InputField(desc="summary of current coverage and gaps")
    missing_entities: str = dspy.InputField(desc="entities or topics that need more coverage")
    previous_queries: str = dspy.InputField(desc="queries used in previous hops")
    query: str = dspy.OutputField(desc="a focused search query targeting missing information")
    reasoning: str = dspy.OutputField(desc="explanation of what this query aims to find")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with adaptive retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.coverage_assessor = dspy.ChainOfThought(CoverageAssessment)
        self.adaptive_query_gen = dspy.ChainOfThought(AdaptiveQueryGenerator)
        self.initial_k = 10
        self.max_hops = 3

    def forward(self, claim):
        # Track unique documents by title and store with metadata
        seen_titles = set()
        all_docs_with_scores = []  # List of (doc, score, hop) tuples
        previous_queries = [claim]

        # HOP 1: Initial broad retrieval
        hop1_retrieval = dspy.Retrieve(k=self.initial_k)(claim)
        hop1_docs = hop1_retrieval.passages

        # Extract titles and store unique documents
        for doc in hop1_docs:
            title = doc.split(" | ")[0] if " | " in doc else doc[:100]
            if title not in seen_titles:
                seen_titles.add(title)
                # ColBERT scores are typically available; default to 1.0 if not
                all_docs_with_scores.append((doc, 1.0, 1))

        # Assess coverage after hop 1
        passages_text = "\n\n".join([doc for doc, _, _ in all_docs_with_scores[:10]])
        coverage = self.coverage_assessor(
            claim=claim,
            retrieved_passages=passages_text
        )

        confidence = float(coverage.confidence_score) if isinstance(coverage.confidence_score, (int, float)) else 0.5

        # HOP 2: Adaptive retrieval based on coverage
        if confidence < 0.9:  # Continue if not highly confident
            # Determine adaptive k
            if confidence < 0.3:
                hop2_k = 10
            elif confidence < 0.6:
                hop2_k = 8
            else:
                hop2_k = 5

            # Generate adaptive query targeting missing entities
            missing_entities_str = ", ".join(coverage.missing_entities) if coverage.missing_entities else "additional context"
            hop2_query_result = self.adaptive_query_gen(
                claim=claim,
                coverage_summary=coverage.coverage_summary,
                missing_entities=missing_entities_str,
                previous_queries="; ".join(previous_queries)
            )
            hop2_query = hop2_query_result.query
            previous_queries.append(hop2_query)

            # Retrieve with adaptive k
            hop2_retrieval = dspy.Retrieve(k=hop2_k)(hop2_query)
            hop2_docs = hop2_retrieval.passages

            # Store unique documents from hop 2
            for doc in hop2_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc[:100]
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs_with_scores.append((doc, 0.9, 2))

            # Reassess coverage after hop 2
            passages_text = "\n\n".join([doc for doc, _, _ in all_docs_with_scores[:15]])
            coverage = self.coverage_assessor(
                claim=claim,
                retrieved_passages=passages_text
            )
            confidence = float(coverage.confidence_score) if isinstance(coverage.confidence_score, (int, float)) else 0.5

        # HOP 3: Final retrieval if still needed
        if confidence < 0.9 and len(all_docs_with_scores) < 21:
            # Determine adaptive k for hop 3
            if confidence < 0.3:
                hop3_k = 10
            elif confidence < 0.6:
                hop3_k = 8
            else:
                hop3_k = 5

            # Generate final adaptive query
            missing_entities_str = ", ".join(coverage.missing_entities) if coverage.missing_entities else "remaining context"
            hop3_query_result = self.adaptive_query_gen(
                claim=claim,
                coverage_summary=coverage.coverage_summary,
                missing_entities=missing_entities_str,
                previous_queries="; ".join(previous_queries)
            )
            hop3_query = hop3_query_result.query
            previous_queries.append(hop3_query)

            # Retrieve with adaptive k
            hop3_retrieval = dspy.Retrieve(k=hop3_k)(hop3_query)
            hop3_docs = hop3_retrieval.passages

            # Store unique documents from hop 3
            for doc in hop3_docs:
                title = doc.split(" | ")[0] if " | " in doc else doc[:100]
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs_with_scores.append((doc, 0.8, 3))

        # Ensure exactly 21 documents
        final_docs = [doc for doc, _, _ in all_docs_with_scores]

        if len(final_docs) < 21:
            # Pad with additional retrievals if needed
            padding_needed = 21 - len(final_docs)
            padding_retrieval = dspy.Retrieve(k=padding_needed * 2)(claim)
            for doc in padding_retrieval.passages:
                title = doc.split(" | ")[0] if " | " in doc else doc[:100]
                if title not in seen_titles and len(final_docs) < 21:
                    final_docs.append(doc)

        # Truncate to exactly 21 if we have more
        final_docs = final_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
