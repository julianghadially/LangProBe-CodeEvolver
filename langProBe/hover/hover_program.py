import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ParallelQueryGenerator(dspy.Signature):
    """Generate 3 diverse queries from the claim, each targeting different semantic aspects to maximize information coverage."""

    claim: str = dspy.InputField(desc="The claim to verify")
    query1: str = dspy.OutputField(desc="Query focusing on entities, actors, people, or organizations mentioned")
    query2: str = dspy.OutputField(desc="Query focusing on relationships, actions, events, or processes")
    query3: str = dspy.OutputField(desc="Query focusing on temporal context, locations, or background details")


class CoverageAnalyzer(dspy.Signature):
    """Analyze the claim and current documents to identify 2 distinct missing information angles that would help verify the claim."""

    claim: str = dspy.InputField(desc="The claim to verify")
    current_docs: str = dspy.InputField(desc="Summary of documents retrieved so far")
    missing_angle1: str = dspy.OutputField(desc="First specific information gap or angle to explore")
    missing_angle2: str = dspy.OutputField(desc="Second distinct information gap or angle to explore")


class UtilityReranker(dspy.Signature):
    """Score each document on its relevance and utility for verifying the claim. Return a comma-separated list of document titles in descending order of utility."""

    claim: str = dspy.InputField(desc="The claim to verify")
    documents: str = dspy.InputField(desc="List of unique documents with titles")
    ranked_titles: str = dspy.OutputField(desc="Comma-separated list of document titles, ordered from most to least useful")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using parallel diverse query expansion.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15  # Retrieve k=15 for each of 3 searches
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Stage 1: Generate diverse queries in parallel
        self.query_generator = dspy.ChainOfThought(ParallelQueryGenerator)

        # Stage 2: Analyze coverage and generate adaptive queries
        self.coverage_analyzer = dspy.ChainOfThought(CoverageAnalyzer)

        # Stage 3: Rerank by utility
        self.reranker = dspy.ChainOfThought(UtilityReranker)

    def _extract_title(self, doc: str) -> str:
        """Extract document title from passage string (format: 'title | content')."""
        return doc.split(" | ")[0] if " | " in doc else doc.split("\n")[0]

    def _deduplicate_docs(self, docs: list[str]) -> list[str]:
        """Remove duplicate documents based on title."""
        seen_titles = set()
        unique_docs = []
        for doc in docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        return unique_docs

    def forward(self, claim):
        # STAGE 1: Diverse Query Generation (1 search)
        # Generate 3 diverse queries targeting different semantic aspects
        queries = self.query_generator(claim=claim)

        # Retrieve documents for the first query only
        stage1_docs = self.retrieve_k(queries.query1).passages

        # Track unique documents by title
        all_docs = stage1_docs.copy()
        unique_docs = self._deduplicate_docs(all_docs)

        # STAGE 2: Adaptive Expansion (2 searches)
        # Analyze coverage to identify missing information
        current_summary = "\n".join([f"- {self._extract_title(doc)}" for doc in unique_docs[:10]])
        coverage = self.coverage_analyzer(claim=claim, current_docs=current_summary)

        # Generate and execute 2 additional queries based on coverage analysis
        stage2_query1 = coverage.missing_angle1
        stage2_query2 = coverage.missing_angle2

        stage2_docs1 = self.retrieve_k(stage2_query1).passages
        stage2_docs2 = self.retrieve_k(stage2_query2).passages

        # Combine all documents and deduplicate
        all_docs.extend(stage2_docs1)
        all_docs.extend(stage2_docs2)
        unique_docs = self._deduplicate_docs(all_docs)

        # STAGE 3: Fusion & Reranking
        # If we have more than 21 documents, rerank by utility
        if len(unique_docs) > 21:
            # Format documents for reranking
            doc_list = "\n".join([f"{i+1}. {self._extract_title(doc)}: {doc[:200]}..."
                                  for i, doc in enumerate(unique_docs)])

            # Get ranked titles
            reranked = self.reranker(claim=claim, documents=doc_list)
            ranked_titles = [t.strip() for t in reranked.ranked_titles.split(",")]

            # Reorder documents based on ranking
            title_to_doc = {self._extract_title(doc): doc for doc in unique_docs}
            final_docs = []
            for title in ranked_titles:
                if title in title_to_doc and len(final_docs) < 21:
                    final_docs.append(title_to_doc[title])

            # If ranking didn't cover all, append remaining docs up to 21
            for doc in unique_docs:
                if len(final_docs) >= 21:
                    break
                if doc not in final_docs:
                    final_docs.append(doc)
        else:
            # If 21 or fewer, use all unique documents
            final_docs = unique_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
