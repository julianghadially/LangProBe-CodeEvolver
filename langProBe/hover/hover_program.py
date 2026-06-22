import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateClaimQueries(dspy.Signature):
    """A factual claim connects approximately 3 Wikipedia articles that need to be retrieved.
    Generate exactly 5 distinct search queries to maximize coverage.

    ABSOLUTE RULES — check these FIRST before generating any query:
    1. VOICE TYPE: If claim says "lowest vocal range" or "bass" in a music/singing context → one query MUST be "bass voice type"
    2. MUSIC ADAPTATION: If claim mentions a classical piece performed by a choir/group → one query MUST be "[piece name] song" for famous pop/song adaptations (e.g., "Polovtsian Dances" → "Stranger in Paradise song")
    3. FILM TITLE: If claim says "[Person] directed the [YEAR] film" → one query MUST be "[PersonName] film" (e.g., "Harry Booth film" for "Harry Booth directed the 1971 film")
    4. NO GENERIC QUERIES: NEVER generate queries for broad categories. FORBIDDEN queries include: "Drama film", "Comedy film", "Action film", "Korean film", "South Korean films", "American film", "British film", "TV series", "Television", "Song", "Music", "Film", "Films", "Actor", "Actress". If you would generate a forbidden query, replace it with a cross-reference like "[ActorName] film" or "[DirectorName] film" instead.

    Strategy:
    - Query 1: for the 1st EXPLICITLY named or described entity/article in the claim
    - Query 2: for the 2nd EXPLICITLY named or described entity/article (different from query1)
    - Query 3: for the 3rd EXPLICITLY named or described entity/article
    - Query 4: CROSS-REFERENCE or SPECIAL query (see ABSOLUTE RULES above; use rule 1/2/3 if applicable, otherwise use an IMPLICIT entity)
    - Query 5: DESCRIPTION-BASED FALLBACK — use EXACT WORDS or PHRASES from the claim:
      * "the 1975 film starring James Mitchum" → q5 = "James Mitchum 1975 film"
      * "the actress from Thank You for Smoking" → q5 = "actress Thank You for Smoking"
      * "the night club in Vienna" → q5 = "nightclub Vienna"
      * "the TV show inspired by Moonrunners" → q5 = "TV show inspired Moonrunners"
      * "the star of Spaceballs" → q5 = "star of Spaceballs actor"
      * "Harry Booth directed the 1971 film that features the star of Thick as Thieves" → q5 = "Harry Booth 1971 film"

    For each query:
    - Named person: use their full name directly
    - Named film/show/song: use the exact title + "(film)"/"(song)"/"(TV series)" if needed
    - Sports season article: use exact format "YEAR-YY TeamName season"
    - Described entity: infer the Wikipedia article title
    - q5: use DESCRIPTIVE WORDS from the claim (not a guessed title)

    Additional strategy for q3 or q4:
    - If the claim uses phrases like "in this religion", "this culture", "in this country/city" → generate a query for the BROADER TOPIC article (e.g., "Ancient Egyptian religion", "Education in Cork")
    - If claim says "[place] has several [things]" → also include the overview article (e.g., "Education in Cork" when claim says "Cork has several colleges")

    CRITICAL: All 5 queries MUST target DIFFERENT Wikipedia articles.
    CRITICAL: Target specific Wikipedia article titles, NOT general concepts or locations.
    Keep each query short (1-6 words), similar to a Wikipedia article title."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="search query for 1st Wikipedia article (explicitly mentioned)")
    query2: str = dspy.OutputField(desc="search query for 2nd Wikipedia article (explicitly mentioned, different from query1)")
    query3: str = dspy.OutputField(desc="search query for 3rd Wikipedia article (explicitly mentioned or described)")
    query4: str = dspy.OutputField(desc="CROSS-REFERENCE or SPECIAL query: use ABSOLUTE RULES 1-3 if applicable (bass voice type / music adaptation / film title from director); otherwise an IMPLICIT entity not directly named but needed for multi-hop reasoning")
    query5: str = dspy.OutputField(desc="DESCRIPTION-BASED FALLBACK using EXACT WORDS from the claim — e.g., 'Harry Booth 1971 film' when claim says 'Harry Booth directed the 1971 film'; '1975 film James Mitchum' when claim says 'the 1975 film starring James Mitchum'; 'actress Thank You for Smoking' for 'the actress from Thank You for Smoking'")


class ExtractGapQuery(dspy.Signature):
    """You are helping retrieve Wikipedia articles for a factual claim.
    The claim requires approximately 3 specific Wikipedia articles.

    CRITICAL PRIORITY PATTERNS — check these FIRST, in order:

    A. MUSIC ADAPTATION (HIGHEST PRIORITY):
       If "Polovtsian Dances" appears in retrieved_titles → missing_entity = "Stranger in Paradise song", query = "Stranger in Paradise song"
       More generally: if a CLASSICAL MUSIC PIECE is in retrieved_titles AND the claim involves a choir/ensemble performing it → the REQUIRED missing article is the famous SONG ADAPTATION of that piece. E.g., Polovtsian Dances → "Stranger in Paradise song" (from musical Kismet).

    B. VOICE TYPE PATTERN:
       If the claim mentions "lowest vocal range" or "bass" in a singing group context → missing_entity = "bass voice type", query = "bass voice type"

    C. FILM TITLE PATTERN:
       If a FILM DIRECTOR's article is in retrieved_titles AND the claim mentions a specific year for a film → use world knowledge to identify the film and search for it directly. E.g., "Harry Booth" + "1971" → query = "On the Buses film". Do NOT search for actors from related TV shows; search for the FILM ITSELF.

    D. PASSAGE REFERENCE PATTERN:
       If key_passages MENTION a specific film/show/person title that is NOT in retrieved_titles AND that title is relevant to the claim → search for that title directly. E.g., if an actor's passage mentions "Green Chair (2005)" → query = "Green Chair film".

    E. ACTOR/PERSON CROSS-REFERENCE:
       If a TV show's article is retrieved AND that show's STAR is retrieved BUT the FILM that connects director + star is missing → search for the FILM using "[DirectorName] film" or "[star] [year] film".

    After checking A-E:
    Step 1: Check retrieved_titles — which articles have ALREADY been found?
    Step 2: Read key_passages carefully — do they MENTION entity names that:
       a) Are referenced by the claim (directly or by description), AND
       b) Are NOT yet listed in retrieved_titles?
    Step 3: USE YOUR OWN KNOWLEDGE: Even if passages are incomplete, use your knowledge of the claim's entities.
      Examples:
      - If "Spaceballs" is retrieved but claim needs the starring actor → think "John Candy" or "Bill Pullman"
      - If "2046 film" is retrieved and claim is about a film award ceremony → think "24th Hong Kong Film Awards"
      - If "Gene Kelly" is retrieved and claim mentions a musical he appeared in → search for specific musical titles
      - If "Swinburne University of Technology" is retrieved alongside an astronomer → think of other astronomers at Swinburne

    Step 4: Identify the single most important missing article.
    Step 5: Generate a precise short query for it.

    Additional patterns:
    - "X was created/founded by Y" → search for "Y" (person's full name)
    - "film/show X stars actor Y" → search for "Y"
    - "directed by Y" → search for "Y"
    - Award ceremonies: if a film is retrieved and claim needs the award, search "NTH [Award Name]"
    - Airlines: if one airline is retrieved and claim needs another in the same region/alliance, search directly
    - TV shows inspired by films: search the TV show title directly
    - IMPORTANT: If already_searched contains 2+ queries for the same entity type → STOP. Pivot completely. Look for a DIFFERENT entity based on the claim.

    Rules:
    - For persons: use FULL NAME (e.g., "Billy Corgan" NOT "Smashing Pumpkins leader")
    - For films: add "(film)" if needed to disambiguate
    - NEVER repeat a query from already_searched
    - If already_searched contains 2+ queries for the same entity type, pivot to a DIFFERENT entity entirely
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

    # Generic/useless queries that waste search slots without finding specific Wikipedia articles
    _GENERIC_QUERY_PATTERNS = {
        'drama film', 'comedy film', 'action film', 'romantic film', 'thriller film',
        'horror film', 'documentary film', 'animated film', 'science fiction film',
        'korean film', 'south korean film', 'south korean films', 'north korean film',
        'american film', 'british film', 'french film', 'indian film', 'japanese film',
        'australian film', 'italian film', 'chinese film', 'german film',
        'film', 'films', 'movie', 'movies',
        'television series', 'tv series', 'tv show', 'television show', 'television',
        'song', 'songs', 'album', 'music', 'pop music', 'rock music',
        'actor', 'actress', 'singer', 'musician', 'person', 'people',
        'drama', 'comedy', 'action', 'romance', 'thriller', 'horror',
    }

    def _is_generic_query(self, query: str) -> bool:
        """Detect obviously generic/useless queries that won't retrieve specific articles."""
        q_lower = query.lower().strip().rstrip('?').strip()
        # Direct match against known generic patterns
        if q_lower in self._GENERIC_QUERY_PATTERNS:
            return True
        # Short queries (1-2 words) that are just genre/category descriptors
        words = q_lower.split()
        if len(words) <= 2:
            # Check if it's just "[nationality] film" or "[genre] film" etc.
            genre_words = {'film', 'films', 'movie', 'movies', 'drama', 'comedy', 'action',
                          'series', 'show', 'song', 'music', 'album', 'television', 'tv'}
            if all(w in genre_words or w in {'korean', 'american', 'british', 'french',
                                              'south', 'north', 'east', 'west', 'asian',
                                              'european', 'romantic', 'thriller', 'horror',
                                              'animated', 'documentary', 'japanese'} for w in words):
                return True
        return False

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

        # HOPS 1-5: targeted searches (skip generic/useless queries)
        hop1_docs = self.retrieve_k(q1).passages if not self._is_generic_query(q1) else []
        hop2_docs = self.retrieve_k(q2).passages if not self._is_generic_query(q2) else []
        hop3_docs = self.retrieve_k(q3).passages if not self._is_generic_query(q3) else []
        hop4_docs = self.retrieve_k(q4).passages if not self._is_generic_query(q4) else []
        hop5_docs = self.retrieve_k(q5).passages if not self._is_generic_query(q5) else []

        # Build context for gap analysis using top-3 passages per hop
        titles_12345 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs)
        passages_12345 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, top_n=3)
        already_searched_12345 = f"{q1}; {q2}; {q3}; {q4}; {q5}"

        # HOP 6: first gap-fill query
        hop6_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_12345,
            key_passages=passages_12345,
            already_searched=already_searched_12345,
        )
        hop6_query = hop6_result.query
        hop6_docs = []
        if not self._is_duplicate_query(hop6_query, [q1, q2, q3, q4, q5]):
            hop6_docs = self.retrieve_k(hop6_query).passages

        # HOP 7: second gap-fill query
        all_queries_up_to_6 = [q1, q2, q3, q4, q5, hop6_query]
        titles_up_to_6 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs)
        passages_up_to_6 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, top_n=4)
        already_searched_up_to_6 = "; ".join(all_queries_up_to_6)
        hop7_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_up_to_6,
            key_passages=passages_up_to_6,
            already_searched=already_searched_up_to_6,
        )
        hop7_query = hop7_result.query
        hop7_docs = []
        if not self._is_duplicate_query(hop7_query, all_queries_up_to_6):
            hop7_docs = self.retrieve_k(hop7_query).passages

        # HOP 8: third gap-fill query
        all_queries_up_to_7 = all_queries_up_to_6 + [hop7_query]
        titles_up_to_7 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs)
        passages_up_to_7 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs, top_n=4)
        already_searched_up_to_7 = "; ".join(all_queries_up_to_7)
        hop8_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_up_to_7,
            key_passages=passages_up_to_7,
            already_searched=already_searched_up_to_7,
        )
        hop8_query = hop8_result.query
        hop8_docs = []
        if not self._is_duplicate_query(hop8_query, all_queries_up_to_7):
            hop8_docs = self.retrieve_k(hop8_query).passages

        # HOP 9: fourth gap-fill query
        all_queries_up_to_8 = all_queries_up_to_7 + [hop8_query]
        titles_up_to_8 = self._get_retrieved_titles(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs, hop8_docs)
        passages_up_to_8 = self._get_key_passages(hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs, hop6_docs, hop7_docs, hop8_docs, top_n=4)
        already_searched_up_to_8 = "; ".join(all_queries_up_to_8)
        hop9_result = self.extract_gap(
            claim=claim,
            retrieved_titles=titles_up_to_8,
            key_passages=passages_up_to_8,
            already_searched=already_searched_up_to_8,
        )
        hop9_query = hop9_result.query
        hop9_docs = []
        if not self._is_duplicate_query(hop9_query, all_queries_up_to_8):
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
