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
    """You are helping retrieve Wikipedia articles for a factual claim.
    The claim requires approximately 3 specific Wikipedia articles.

    Step 1: Check retrieved_titles — which articles have ALREADY been found?
    Step 2: Read key_passages carefully — do they MENTION entity names that:
       a) Are referenced by the claim (directly or by description), AND
       b) Are NOT yet listed in retrieved_titles?
    Step 3: Identify the single most important missing article.
    Step 4: Generate a precise short query for it.

    CRITICAL patterns — look for these in key_passages:
    - "X was created/founded by Y" → if claim needs the creator, search for "Y" (person's full name)
    - "film/show X stars actor Y" → if claim needs that actor, search for "Y"
    - "owned/produced by company X" → if claim needs that owner, search for "X"
    - "book/work A written by B" → if claim references the author, search for "B"
    - "directed by Y" → if claim needs the director, search for "Y"

    Rules:
    - For persons: use FULL NAME (e.g., "Billy Corgan" NOT "Smashing Pumpkins leader")
    - For films: add "(film)" if needed to disambiguate
    - NEVER repeat a query from already_searched
    - Target specific Wikipedia article titles, not general topics"""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="pipe-separated list of Wikipedia article titles already retrieved — check these to know what has been found")
    key_passages: str = dspy.InputField(desc="text excerpts from the top retrieved article per hop — scan these for entity names the claim also needs but are missing from retrieved_titles")
    already_searched: str = dspy.InputField(desc="queries already used — do NOT repeat these")
    missing_entity: str = dspy.OutputField(desc="the specific missing Wikipedia article title (e.g., 'Billy Corgan', 'Mars Incorporated', 'Warren Fu')")
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

    def _get_retrieved_titles(self, *doc_lists) -> str:
        """Get unique document titles from multiple hop results as pipe-separated string."""
        titles = []
        seen = set()
        for docs in doc_lists:
            for doc in docs[:7]:  # top 7 from each hop
                title_norm = self._doc_title(doc)
                if title_norm not in seen:
                    seen.add(title_norm)
                    titles.append(doc.split(" | ")[0].strip())  # original casing
        return " | ".join(titles)

    def _get_key_passages(self, *doc_lists, top_n=3) -> str:
        """Get top passages from each hop for gap analysis context."""
        passages = []
        for docs in doc_lists:
            for doc in docs[:top_n]:
                passage = doc[:350]
                passages.append(passage)
        return "\n\n---\n\n".join(passages)

    def _rrf_merge(self, all_hops, k=60, max_docs=21):
        """Reciprocal Rank Fusion merge of multiple retrieval lists."""
        rrf_scores = {}
        doc_map = {}  # normalized_title -> doc string

        for hop_docs in all_hops:
            for rank, doc in enumerate(hop_docs):
                title = self._doc_title(doc)
                if title not in rrf_scores:
                    rrf_scores[title] = 0.0
                    doc_map[title] = doc
                rrf_scores[title] += 1.0 / (rank + k)

        # Sort by RRF score descending
        sorted_titles = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)
        return [doc_map[t] for t in sorted_titles[:max_docs]]

    def forward(self, claim):
        # STEP 1: Generate 3 targeted queries from the claim in one shot
        cq = self.generate_queries(claim=claim)
        q1, q2, q3 = cq.query1, cq.query2, cq.query3

        # HOPS 1-3: targeted searches for each claim entity
        hop1_docs = self.retrieve_k(q1).passages
        hop2_docs = self.retrieve_k(q2).passages
        hop3_docs = self.retrieve_k(q3).passages

        # Build structured context for gap analysis: titles + focused passages
        titles_123 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs)
        passages_123 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs)
        already_searched_123 = f"{q1}; {q2}; {q3}"

        # HOP 4: first gap-fill query
        hop4_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_123,
            key_passages=passages_123,
            already_searched=already_searched_123,
        )
        hop4_query = hop4_result.query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # HOP 5: second gap-fill query
        titles_1234 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs)
        passages_1234 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs)
        already_searched_1234 = f"{already_searched_123}; {hop4_query}"
        hop5_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_1234,
            key_passages=passages_1234,
            already_searched=already_searched_1234,
        )
        hop5_query = hop5_result.query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # HOP 6: third gap-fill query (only if not a duplicate)
        all_queries = [q1, q2, q3, hop4_query, hop5_query]
        titles_12345 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs)
        passages_12345 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs)
        already_searched_12345 = f"{already_searched_1234}; {hop5_query}"
        hop6_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_12345,
            key_passages=passages_12345,
            already_searched=already_searched_12345,
        )
        hop6_query = hop6_result.query
        hop6_docs = []
        if not self._is_duplicate_query(hop6_query, all_queries):
            hop6_docs = self.retrieve_k(hop6_query).passages

        # RRF merge: includes docs ranked 4+ if they score well overall
        all_hops_to_merge = [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs]
        if hop6_docs:
            all_hops_to_merge.append(hop6_docs)

        final_docs = self._rrf_merge(all_hops_to_merge, k=60, max_docs=21)
        return dspy.Prediction(retrieved_docs=final_docs[:21])
