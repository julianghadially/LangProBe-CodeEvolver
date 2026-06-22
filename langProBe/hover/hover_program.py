import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateClaimQueries(dspy.Signature):
    """A factual claim connects approximately 3 Wikipedia articles that need to be retrieved.
    Generate exactly 5 distinct search queries to maximize coverage.

    Strategy:
    - Query 1: for the 1st EXPLICITLY named or described entity/article in the claim
    - Query 2: for the 2nd EXPLICITLY named or described entity/article (different from query1)
    - Query 3: for the 3rd EXPLICITLY named or described entity/article
    - Query 4: for an entity that is IMPLICIT or INFERRED — not directly named in the claim,
      but likely needed given the multi-hop reasoning structure. Examples:
      * If claim says "X directed Y" and Y is a film → q4 might target the film studio or a co-star
      * If claim says "X was at university Z" → q4 might target a notable person associated with Z
      * If claim says "X appeared in Y" → q4 might target the director or creator of Y
      * If the claim's logic requires an intermediate hop entity → q4 targets that entity
    - Query 5: a DESCRIPTION-BASED FALLBACK — use EXACT WORDS or PHRASES from the claim to
      search for an entity that is described but not directly named. Use the claim's own language.
      Examples:
      * Claim says "the 1975 film starring James Mitchum" → q5 = "James Mitchum 1975 film"
      * Claim says "the actress from Thank You for Smoking" → q5 = "actress Thank You for Smoking"
      * Claim says "the night club in Vienna" → q5 = "nightclub Vienna electronic music"
      * Claim says "the TV show inspired by Moonrunners" → q5 = "TV show inspired Moonrunners"
      * Claim says "the star of Spaceballs" → q5 = "star of Spaceballs actor"
      * Claim says "a film directed by the same director as X" → q5 = "director X film"

    For each query:
    - Named person: use their full name directly
    - Named film/show/song: use the exact title + "(film)"/"(song)"/"(TV series)" if needed
    - Sports season article: use exact format "YEAR-YY TeamName season"
    - Described entity: infer the Wikipedia article title
    - q5: use DESCRIPTIVE WORDS from the claim (not a guessed title)

    Additional strategy for q3 or q4:
    - If the claim uses phrases like "in this religion", "this culture", "in this country/city" → generate a query for the BROADER TOPIC article (e.g., "Ancient Egyptian religion", "Education in Cork")
    - If the claim mentions "[a composition/piece] was performed by X" → check if the composition has a famous adaptation (e.g., "Stranger in Paradise (song)" from "Polovtsian Dances")
    - If claim says "[place] has several [things]" → also include the overview article (e.g., "Education in Cork" when claim says "Cork has several colleges")

    CRITICAL: All 5 queries MUST target DIFFERENT Wikipedia articles.
    CRITICAL: Target specific Wikipedia article titles, NOT general concepts or locations.
    Keep each query short (1-6 words), similar to a Wikipedia article title."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="search query for 1st Wikipedia article (explicitly mentioned)")
    query2: str = dspy.OutputField(desc="search query for 2nd Wikipedia article (explicitly mentioned, different from query1)")
    query3: str = dspy.OutputField(desc="search query for 3rd Wikipedia article (explicitly mentioned or described)")
    query4: str = dspy.OutputField(desc="search query for 4th Wikipedia article — an IMPLICIT or INFERRED entity not directly named in the claim but needed for multi-hop reasoning")
    query5: str = dspy.OutputField(desc="search query using DESCRIPTIVE WORDS from the claim for an entity that is described but not directly named — use phrases from the claim itself (e.g., '1975 film James Mitchum' when claim says 'the 1975 film starring James Mitchum'; 'actress Thank You for Smoking' for 'the actress from Thank You for Smoking'; 'nightclub Vienna' for 'the nightclub in Vienna')")


class ExtractGapQueries(dspy.Signature):
    """You are helping retrieve Wikipedia articles for a factual claim.
    The claim requires approximately 3 specific Wikipedia articles.

    Step 1: Check retrieved_titles — which articles have ALREADY been found?
    Step 2: Read key_passages carefully — do they MENTION entity names that:
       a) Are referenced by the claim (directly or by description), AND
       b) Are NOT yet listed in retrieved_titles?
    Step 3: USE YOUR OWN KNOWLEDGE: Even if passages are incomplete, use your knowledge
      of the claim's entities to identify what Wikipedia articles are required.
      Examples:
      - If "Spaceballs" is retrieved but claim needs the starring actor → think "John Candy" or "Bill Pullman"
      - If "Liza Minnelli discography" is retrieved → think about films she appeared in early career
      - If "Comair (South Africa)" is retrieved and claim mentions a British Airways franchise → think "British Airways franchise destinations"
      - If "2046 film" is retrieved and claim is about a film award ceremony → think "24th Hong Kong Film Awards"
      - If "airBaltic" is retrieved and claim is about another Baltic airline → think "Air Lituanica"
      - If "Gene Kelly" is retrieved and claim mentions a musical he appeared in → search for specific musical titles

    Step 4: Identify the TWO most important missing articles.
    Step 5: Generate precise short queries for each.

    CRITICAL patterns (look in key_passages AND use world knowledge):
    - "X was created/founded by Y" → search for "Y" (person's full name)
    - "film/show X stars actor Y" → search for "Y"
    - "owned/produced by company X" → search for "X"
    - "directed by Y" → search for "Y"
    - Award ceremonies: if a film is retrieved and claim needs the award, search "NTH [Award Name]" (e.g., "24th Hong Kong Film Awards")
    - Airlines: if one airline is retrieved and claim needs another in the same region/alliance, search the other airline directly
    - TV shows inspired by films: search the TV show title directly (e.g., "The Dukes of Hazzard")
    - Music adaptations: if a classical piece is retrieved, check if it has a famous pop adaptation

    Rules:
    - Generate EXACTLY 2 queries for 2 DIFFERENT missing entities
    - NEVER repeat any query from already_searched
    - If only 1 entity appears missing, use world knowledge for the second query (creative alternative angle or related article)
    - For persons: use FULL NAME
    - For films: add "(film)" if needed
    - Target specific Wikipedia article titles, not general topics
    - If already_searched contains 2+ queries for same entity type, pivot to completely different angle"""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="pipe-separated list of Wikipedia article titles already retrieved — check these to know what has been found")
    key_passages: str = dspy.InputField(desc="text excerpts from the top retrieved article per hop — scan these for entity names the claim also needs but are missing from retrieved_titles")
    already_searched: str = dspy.InputField(desc="queries already used — do NOT repeat these")
    missing_entity_1: str = dspy.OutputField(desc="the first missing Wikipedia article title needed by the claim")
    query1: str = dspy.OutputField(desc="short search query (1-6 words) for the first missing article")
    missing_entity_2: str = dspy.OutputField(desc="the second missing Wikipedia article title (DIFFERENT from missing_entity_1)")
    query2: str = dspy.OutputField(desc="short search query (1-6 words) for the second missing article — MUST be different from query1 and from already_searched")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program."""

    def __init__(self):
        super().__init__()
        self.k = 25
        self.generate_queries = dspy.ChainOfThought(GenerateClaimQueries)
        self.extract_gap = dspy.ChainOfThought(ExtractGapQueries)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def _is_duplicate_query(self, new_query: str, previous_queries: list) -> bool:
        """Check if new_query is effectively the same as any previous query."""
        from difflib import SequenceMatcher
        new_norm = new_query.lower().strip().rstrip('?').strip()
        for prev in previous_queries:
            prev_norm = prev.lower().strip().rstrip('?').strip()
            if new_norm == prev_norm:
                return True
            # Fuzzy check: catch near-duplicates (typos, formatting variations)
            similarity = SequenceMatcher(None, new_norm, prev_norm).ratio()
            if similarity > 0.85:
                return True
        return False

    def _get_retrieved_titles(self, *doc_lists) -> str:
        """Get unique document titles from multiple hop results as pipe-separated string."""
        titles = []
        seen = set()
        for docs in doc_lists:
            for doc in docs[:10]:  # top 10 from each hop
                title_norm = self._doc_title(doc)
                if title_norm not in seen:
                    seen.add(title_norm)
                    titles.append(doc.split(" | ")[0].strip())  # original casing
        return " | ".join(titles)

    def _get_key_passages(self, *doc_lists, top_n=3) -> str:
        """Get the top passage from each hop, truncated for focus."""
        passages = []
        for docs in doc_lists:
            for doc in docs[:top_n]:
                passage = doc[:1200]
                passages.append(passage)
        return "\n\n---\n\n".join(passages)

    def _score_based_merge(self, all_hops, max_docs=21):
        """Score-based merge using inverse-rank scoring across all hops.

        Each doc gets score = sum of 1/(rank+1) across all hops that returned it.
        Docs retrieved by multiple hops AND ranked higher get priority.
        """
        doc_scores: dict = {}  # normalized_title -> (total_score, first_doc_text)
        for hop_docs in all_hops:
            for rank, doc in enumerate(hop_docs):
                title = self._doc_title(doc)
                score = 1.0 / (rank + 1)
                if title in doc_scores:
                    doc_scores[title] = (doc_scores[title][0] + score, doc_scores[title][1])
                else:
                    doc_scores[title] = (score, doc)

        # Sort by score descending, take top max_docs
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [doc for _, (_, doc) in sorted_items[:max_docs]]

    def forward(self, claim):
        # STEP 1: Generate 5 targeted queries from the claim in one shot
        cq = self.generate_queries(claim=claim)
        q1, q2, q3, q4, q5 = cq.query1, cq.query2, cq.query3, cq.query4, cq.query5

        # HOPS 1-5: targeted searches (3 explicit + 1 inferred + 1 description-based)
        hop1_docs = self.retrieve_k(q1).passages
        hop2_docs = self.retrieve_k(q2).passages
        hop3_docs = self.retrieve_k(q3).passages
        hop4_docs = self.retrieve_k(q4).passages
        hop5_docs = self.retrieve_k(q5).passages

        # Build context for gap analysis using top-3 passages per hop
        titles_12345 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs)
        passages_12345 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, top_n=3)
        already_searched_12345 = f"{q1}; {q2}; {q3}; {q4}; {q5}"

        # HOP 6+7: first dual gap-fill round
        gap1_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_12345,
            key_passages=passages_12345,
            already_searched=already_searched_12345,
        )
        hop6_query = gap1_result.query1
        hop7_query = gap1_result.query2
        hop6_docs = []
        hop7_docs = []
        if not self._is_duplicate_query(hop6_query, [q1, q2, q3, q4, q5]):
            hop6_docs = self.retrieve_k(hop6_query).passages
        if not self._is_duplicate_query(hop7_query, [q1, q2, q3, q4, q5, hop6_query]):
            hop7_docs = self.retrieve_k(hop7_query).passages

        # HOP 8+9: second dual gap-fill round (uses results from all previous hops)
        all_queries_up_to_7 = [q1, q2, q3, q4, q5, hop6_query, hop7_query]
        titles_up_to_7 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs)
        passages_up_to_7 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs, top_n=4)
        already_searched_up_to_7 = "; ".join(all_queries_up_to_7)

        gap2_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_up_to_7,
            key_passages=passages_up_to_7,
            already_searched=already_searched_up_to_7,
        )
        hop8_query = gap2_result.query1
        hop9_query = gap2_result.query2
        hop8_docs = []
        hop9_docs = []
        if not self._is_duplicate_query(hop8_query, all_queries_up_to_7):
            hop8_docs = self.retrieve_k(hop8_query).passages
        if not self._is_duplicate_query(hop9_query, all_queries_up_to_7 + [hop8_query]):
            hop9_docs = self.retrieve_k(hop9_query).passages

        # Collect all non-empty hop results
        all_hops = [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs]
        if hop6_docs:
            all_hops.append(hop6_docs)
        if hop7_docs:
            all_hops.append(hop7_docs)
        if hop8_docs:
            all_hops.append(hop8_docs)
        if hop9_docs:
            all_hops.append(hop9_docs)

        # Score-based merge: inverse-rank scoring, top-21
        final_docs = self._score_based_merge(all_hops, max_docs=21)

        return dspy.Prediction(retrieved_docs=final_docs[:21])
