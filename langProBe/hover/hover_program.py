import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateClaimQueries(dspy.Signature):
    """A factual claim connects approximately 3 Wikipedia articles that need to be retrieved.
    Generate exactly 4 distinct search queries to maximize coverage.

    Strategy:
    - Queries 1-3: one for each EXPLICITLY named or described entity/article in the claim
    - Query 4: for an entity that is IMPLICIT or INFERRED — not directly named in the claim,
      but likely needed given the multi-hop reasoning structure. Examples:
      * If claim says "X directed Y" and Y is a film → q4 might target the film studio or a co-star
      * If claim says "X was at university Z" → q4 might target a notable person associated with Z
      * If claim says "X appeared in Y" → q4 might target the director or creator of Y
      * If the claim's logic requires an intermediate hop entity → q4 targets that entity

    For each query:
    - Named person: use their full name directly
    - Named film/show/song: use the exact title + "(film)"/"(song)"/"(TV series)" if needed
    - Sports season article: use exact format "YEAR-YY TeamName season"
    - Described entity: infer the Wikipedia article title
    - Inferred entity (q4): reason carefully about what intermediate Wikipedia article connects the claim

    CRITICAL: All 4 queries MUST target DIFFERENT Wikipedia articles.
    CRITICAL: Target specific Wikipedia article titles, NOT general concepts or locations.
    Keep each query short (1-6 words), similar to a Wikipedia article title."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="search query for 1st Wikipedia article (explicitly mentioned)")
    query2: str = dspy.OutputField(desc="search query for 2nd Wikipedia article (explicitly mentioned, different from query1)")
    query3: str = dspy.OutputField(desc="search query for 3rd Wikipedia article (explicitly mentioned or described)")
    query4: str = dspy.OutputField(desc="search query for 4th Wikipedia article — an IMPLICIT or INFERRED entity not directly named in the claim but needed for multi-hop reasoning")


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

    def _get_key_passages(self, *doc_lists, top_n=1) -> str:
        """Get the top passage from each hop, truncated for focus."""
        passages = []
        for docs in doc_lists:
            for doc in docs[:top_n]:
                passage = doc[:500]
                passages.append(passage)
        return "\n\n---\n\n".join(passages)

    def forward(self, claim):
        # STEP 1: Generate 4 targeted queries from the claim in one shot
        cq = self.generate_queries(claim=claim)
        q1, q2, q3, q4 = cq.query1, cq.query2, cq.query3, cq.query4

        # HOPS 1-4: targeted searches (3 explicit + 1 inferred)
        hop1_docs = self.retrieve_k(q1).passages
        hop2_docs = self.retrieve_k(q2).passages
        hop3_docs = self.retrieve_k(q3).passages
        hop4_docs = self.retrieve_k(q4).passages

        # Build context for gap analysis using top-1 passage per hop
        titles_1234 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs)
        passages_1234 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs)
        already_searched_1234 = f"{q1}; {q2}; {q3}; {q4}"

        # HOP 5: first gap-fill query
        hop5_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_1234,
            key_passages=passages_1234,
            already_searched=already_searched_1234,
        )
        hop5_query = hop5_result.query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # HOP 6: second gap-fill query (only if not a duplicate)
        all_queries = [q1, q2, q3, q4, hop5_query]
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

        # Interleaved round-robin merge with deduplication
        all_hops = [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs]
        if hop6_docs:
            all_hops.append(hop6_docs)

        seen_titles: set = set()
        final_docs: list = []
        for i in range(self.k):
            for hop_docs in all_hops:
                if i < len(hop_docs) and len(final_docs) < 21:
                    title = self._doc_title(hop_docs[i])
                    if title not in seen_titles:
                        seen_titles.add(title)
                        final_docs.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=final_docs[:21])
