import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateClaimQueries(dspy.Signature):
    """A factual claim mentions approximately 3 Wikipedia articles that need to be retrieved.
    Generate exactly 3 distinct search queries, one for each expected Wikipedia article.

    For each query:
    - Named person (e.g., "Billy Corgan", "Lavinia Greenlaw"): use their full name directly
    - Named film/show/song: use the exact title + "(film)"/"(song)"/"(TV series)" if needed
    - Sports season article: use exact format "YEAR-YY TeamName season" (e.g., "2004-05 Memphis Grizzlies season", "1974-75 New York Islanders season")
    - Described entity (e.g., "the show she starred in"): infer the Wikipedia article title
    - Implied entity: reason to the Wikipedia article title

    CRITICAL: All 3 queries MUST target DIFFERENT Wikipedia articles.
    CRITICAL: Target specific Wikipedia article titles, NOT general concepts or locations.
    Keep each query short (1-6 words), similar to a Wikipedia article title."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="search query for 1st Wikipedia article")
    query2: str = dspy.OutputField(desc="search query for 2nd Wikipedia article (different from query1)")
    query3: str = dspy.OutputField(desc="search query for 3rd Wikipedia article (different from query1 and query2)")


class ExtractGapQuery(dspy.Signature):
    """Given a claim and passages already retrieved, identify the single most important
    Wikipedia article NOT yet retrieved.

    Process:
    1. List the ~3 specific Wikipedia articles required by the claim (persons, films, songs, seasons, etc.)
    2. Check each against retrieved passages to find which are MISSING
    3. Focus on the most important missing article

    Key rules:
    - For persons: search their full name directly (e.g., "Billy Corgan" not "The Smashing Pumpkins")
    - For sports seasons: use exact format "YEAR-YY TeamName season"
    - For films: use film title + "(film)" if needed
    - For songs: use song title + "(song)" if needed
    - NEVER generate a query that is in already_searched (check carefully!)
    - NEVER generate a query for a general concept, location, or topic — target a specific Wikipedia article"""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="passages already retrieved from hops 1-3 (or 1-4 or 1-5)")
    already_searched: str = dspy.InputField(desc="queries already used — do NOT repeat these")
    missing_entity: str = dspy.OutputField(desc="the specific missing Wikipedia article title (e.g., 'Billy Corgan', '2004-05 Memphis Grizzlies season')")
    query: str = dspy.OutputField(desc="short search query (1-6 words) for the missing Wikipedia article")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program."""

    def __init__(self):
        super().__init__()
        self.k = 25
        self.generate_queries = dspy.ChainOfThought(GenerateClaimQueries)
        self.extract_gap = dspy.ChainOfThought(ExtractGapQuery)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def _is_duplicate_query(self, new_query: str, previous_queries: list) -> bool:
        """Check if new_query is effectively the same as any previous query."""
        new_norm = new_query.lower().strip().rstrip('?').strip()
        for prev in previous_queries:
            prev_norm = prev.lower().strip().rstrip('?').strip()
            if new_norm == prev_norm:
                return True
        return False

    def _rrf_merge(self, hop_docs_list: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion: score each doc by sum of 1/(k+rank) across all hops."""
        scores: dict = {}
        doc_by_title: dict = {}

        for hop_docs in hop_docs_list:
            for rank, doc in enumerate(hop_docs):
                title = self._doc_title(doc)
                if title not in scores:
                    scores[title] = 0.0
                    doc_by_title[title] = doc
                scores[title] += 1.0 / (k + rank + 1)

        sorted_titles = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return [doc_by_title[t] for t in sorted_titles[:21]]

    def forward(self, claim):
        # STEP 1: Generate 3 targeted queries from the claim in one shot
        cq = self.generate_queries(claim=claim)
        q1, q2, q3 = cq.query1, cq.query2, cq.query3

        # HOPS 1-3: targeted searches for each claim entity
        hop1_docs = self.retrieve_k(q1).passages
        hop2_docs = self.retrieve_k(q2).passages
        hop3_docs = self.retrieve_k(q3).passages

        # Build context for gap analysis
        context_123 = "\n\n".join(hop1_docs[:3] + hop2_docs[:3] + hop3_docs[:3])
        already_searched_123 = f"{q1}; {q2}; {q3}"

        # HOP 4: first gap-fill query
        hop4_result = self.extract_gap(
            claim=claim,
            passages=context_123,
            already_searched=already_searched_123,
        )
        hop4_query = hop4_result.query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # HOP 5: second gap-fill query
        context_1234 = "\n\n".join(hop1_docs[:2] + hop2_docs[:2] + hop3_docs[:2] + hop4_docs[:2])
        already_searched_1234 = f"{already_searched_123}; {hop4_query}"
        hop5_result = self.extract_gap(
            claim=claim,
            passages=context_1234,
            already_searched=already_searched_1234,
        )
        hop5_query = hop5_result.query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # HOP 6: third gap-fill query
        context_12345 = "\n\n".join(hop1_docs[:2] + hop2_docs[:2] + hop3_docs[:2] + hop4_docs[:2] + hop5_docs[:2])
        already_searched_12345 = f"{already_searched_1234}; {hop5_query}"
        hop6_result = self.extract_gap(
            claim=claim,
            passages=context_12345,
            already_searched=already_searched_12345,
        )
        hop6_query = hop6_result.query
        hop6_docs = []
        if not self._is_duplicate_query(hop6_query, [q1, q2, q3, hop4_query, hop5_query]):
            hop6_docs = self.retrieve_k(hop6_query).passages

        # Reciprocal Rank Fusion merge
        hop_all = [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs]
        if hop6_docs:
            hop_all.append(hop6_docs)
        final_docs = self._rrf_merge(hop_all)

        return dspy.Prediction(retrieved_docs=final_docs[:21])
