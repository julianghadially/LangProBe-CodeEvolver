import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_RETRIEVED_DOCS = 21


class ClaimEntities(dspy.Signature):
    """List every distinct named entity mentioned in the claim — people, works (films, songs,
    albums, books, video games), organizations, places, events, concepts. Output each entity as
    the SPECIFIC name used in the claim, preserving any year, edition or disambiguator
    (e.g. "2005 NASDAQ-100 Open Women's Doubles", "Lolita (1962 film)", "G.I. Joe: Hall of Fame"),
    not a generalized category. Use Wikipedia-style article names.

    A possessive/event phrase that is ITSELF a titled Wikipedia article must be output as the
    FULL phrase verbatim, not just the proper noun inside it:
      - "Douglas MacArthur's escape from the Philippines"  (NOT just "Douglas MacArthur")
      - "Battle of the Bulge", "Treaty of Versailles", "Lincoln's assassination",
        "MacArthur's escape from the Philippines", "2004 United States election".
    A binomial species name "Genus species" (e.g. "Enkianthus campanulatus", "Erythroxylum
    vacciniifolium") means BOTH the species AND its genus have their own article — output BOTH
    the full species name AND the bare genus ("Enkianthus", "Erythroxylum")."""
    claim = dspy.InputField(desc="The claim")
    entities: list[str] = dspy.OutputField(desc="Named entities in the claim, as specific Wikipedia-style names")


class QueryExpansion(dspy.Signature):
    """You are retrieving Wikipedia documents for a claim. This is PURE DOCUMENT RETRIEVAL —
    do NOT verify, fact-check, or judge whether the claim is true, and do NOT decide the claim is
    "supported". Your ONLY job: output search queries for documents STILL MISSING.

    ## What "missing" means (read carefully)
    - An entity is COVERED only if one of the retrieved document TITLES is (or matches) that
      entity. An entity merely MENTIONED inside another article's snippet is NOT covered — it
      still needs its OWN article. (e.g. if an award snippet lists "S. Truett Cathy" as a recipient
      but no retrieved title is "S. Truett Cathy", then "S. Truett Cathy" is missing.)
    - Compare the retrieved TITLES to the claim; any entity the claim depends on that has no
      matching retrieved TITLE is missing.
    - Also compare against the claim's own entities: a claim-named entity with no matching title
      is still missing even if it appears in a snippet.

    ## CRITICAL: mine the snippets for the EXACT proper-noun name
    The retrieved snippets almost always CONTAIN the exact name of the missing entity. Scan every
    snippet for proper nouns the claim depends on, and output a query equal to that EXACT name —
    quoted verbatim from the snippet — not a paraphrase, not a role, not a description.
      - snippet "...directed by Mick Napier"            -> "Mick Napier"   (NOT "Director of Splatter Theatre")
      - snippet "...remake of 2003 Tamil film Dhool"    -> "Dhool"          (NOT "remake of Ranja", NOT a guess like "Sethu")
      - snippet "...ninth season of Deutschland sucht den Superstar" -> "Deutschland sucht den Superstar (season 9)"
      - snippet "...nominated ... in the 24th Hong Kong Film Awards"  -> "24th Hong Kong Film Awards" (NOT "Hong Kong Film Awards")
      - snippet "...won by Ross Case"                    -> "Ross Case"
      - snippet 'including "Punchlines" and ...'         -> "Punchlines"
    Prefer the SPECIFIC disambiguated title a snippet gives (with its year/ordinal/season) over the
    generic version, because the generic version retrieves the wrong/neighbor article.

    ## Mine EVERY proper noun in a snippet, not just the first
    A single work's snippet usually lists MANY names the claim depends on — the full cast, every
    band member, every director, the venue. Output a query for EACH of them, not just the most
    prominent one. Stopping after one or two misses a gold document and zeros the score.
      - A film snippet listing the cast   -> query EVERY named actor (e.g. "Karan Kapoor",
        "Susanna Thompson"), not just the lead.
      - A performer's show snippet         -> mine the VENUE too: "...at the Luxor Las Vegas"
        -> "Luxor Las Vegas"; "...on Broadway at the Neil Simon Theatre" -> "Neil Simon Theatre".
      - An album snippet naming its band   -> the band's article.

    ## Relational patterns: read the NAMED entity's OWN snippet
    When the claim says "the film that X is a remake of", "the director of X", "the host of X",
    "the winner of X", "the band behind X", read the snippet of the article titled X (the entity the
    claim NAMES) — NOT snippets of other, merely-related articles. The original film/host/winner is
    stated in X's own article. Do NOT substitute a different article's answer.

    ## NEVER pre-filter or dismiss a named entity
    This is retrieval, not verification. Do NOT decide an entity is "the wrong country/edition/
    version" and skip it. If a snippet names a proper noun the claim depends on (e.g. a TV show Paul
    Melba appeared on, even if it seems British), OUTPUT it — the article itself is the supporting
    document. An extra low-value query costs almost nothing; a MISSED document zeros the entire
    score. When in doubt, output the query.

    ## Concept / category articles are MISSING documents too
    Concepts (voice types, languages, territory types, taxonomic ranks, genres) have their OWN
    Wikipedia articles. A concept MENTIONED in a snippet or the claim, but with no matching
    retrieved TITLE, is still MISSING — do NOT dismiss it as "just a description" or "already
    covered" because a snippet uses the word. Output its Wikipedia article title as a query:
      - snippet "...the bass, the lowest vocal range..."  -> "Bass (voice type)"
      - claim "...the territory where [Gibraltar]..."      -> "British overseas territories"
      - snippet "...from the Tupi/Guarani word..."         -> BOTH "Tupi language" AND
        "Guarani language" (and "Tupi–Guarani languages")
      - a named species' genus is missing                  -> the genus article ("Enkianthus")
    For a word's etymology, NEVER output just one language: if a snippet names several languages or
    families as the source of a word, query EACH one's article — the gold is whichever matches, and
    you cannot know in advance which.

    ## Map prose references to Wikipedia disambiguated titles
    - "ninth season" -> append "(season 9)"; "season 4" -> "(season 4)".
    - A year/edition ordinal in a snippet is part of the title — keep it exact
      ("24th Hong Kong Film Awards", "1974 Pacific Coast Open", "2046 (film)").
    - People: use their common Wikipedia name ("Mick Napier", "Shim Ji-ho", "Ross Case").

    ## Also use world knowledge
    If a missing entity is implied by the claim but not in any snippet yet (e.g. the director of a
    named film, a cast member, the election a politician lost, the band behind an album, the spouse
    of a person), use your knowledge to NAME it. Still output a proper-noun name, never a role.
    A character and the publication it appears in are DISTINCT articles — output BOTH (e.g. for a
    "comic book character" output the character "Archie Andrews" AND the comic book "Archie comic
    book"; for an album output the album AND the band).

    ## Rules
    - Each query targets ONE entity, is its exact Wikipedia-style title/name, short (1-5 words).
    - Do NOT write full sentences, questions, or justification/verification text.
    - Do NOT repeat a query whose name already matches a retrieved document TITLE, and do not
      repeat any query you can see was already issued (it will be filtered, so pick a NEW entity).
    - Cover DISTINCT missing entities; queries must be non-redundant.
    - NEVER output a role/description ("director of X", "host of X", "remake of X", "winner of X").
      Always output a PROPER-NOUN NAME. If you cannot name it from a snippet or knowledge, SKIP it
      rather than emit a description.
    - When unsure whether an entity is covered, output a query rather than none — a missed document
      zeros the entire score.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    retrieved_docs = dspy.InputField(desc="Documents retrieved so far as '<title> | <snippet>', one per line")
    queries: list[str] = dspy.OutputField(desc="Concise Wikipedia-style search queries — EXACT proper-noun names mined from snippets — for still-missing documents")


class ClaimAnalysis(dspy.Signature):
    """Analyze the claim to identify entities, works, or concepts it refers to but does NOT
    explicitly name — entities that a supporting Wikipedia document would need to cover.

    This is PURE DOCUMENT RETRIEVAL — do NOT verify, fact-check, or judge whether the claim
    is true. Just identify what the claim is referring to and output Wikipedia-style search
    queries for those entities.

    Examples of what to infer (output a PROPER-NOUN NAME, never a description/role):
    - "The actor who starred in an Oscar winning film with Amber Tamblyn" -> "127 Hours"
    - "A genus containing Butein" -> "Dahlia"
    - "The followup novel to the Heir to the Empire trilogy" -> "Vision of the Future"
    - "The director of Pacific Rim" -> "Guillermo del Toro"
    - "The director of Splatter Theatre" -> "Mick Napier"
    - "the film that Ranja is a remake of" -> the original film's name
    - "the season Vanessa Krasniqi took part in" -> "Deutschland sucht den Superstar (season 9)"

    Rules:
    - Each query is a short Wikipedia-style title/name (1-5 words), a PROPER NOUN or a
      well-known concept article title.
    - NEVER output a description or role ("director of X", "host of X", "remake of X"). Always
      output the actual name; if you cannot name it, skip it.
    - Map prose to disambiguated Wikipedia titles ("ninth season" -> "(season 9)"; keep a year or
      ordinal that the claim gives exact).
    - Do NOT repeat entities already explicitly named in the claim.
    - Do NOT verify whether the claim is true. Just identify what it refers to.
    - Infer the Wikipedia article for any CONCEPT / CATEGORY / TYPE the claim depends on, even
      though it is a common noun with no proper-noun name. These have their own articles and are
      MISSING documents:
        - a vocal range ("lowest vocal range" -> "Bass (voice type)"; "highest" -> "Soprano");
        - a territory type ("the territory" with a British monarch -> "British overseas territories");
        - the language a word derives from ("Guarani language", "Tupi language");
        - a taxonomic rank of a named species (the genus/family of "Enkianthus campanulatus"
          -> "Enkianthus").
    - Infer the publication/line a work belongs to. A character and the publication it appears in
      are DISTINCT articles and BOTH may be required: a "comic book character originally written by
      X" -> output BOTH the character article ("Archie Andrews") AND the comic book's article
      ("Archie comic book"); an album -> also the artist's article; a song -> also the album.
    - When unsure, output a query rather than nothing — missing documents cost the entire score.
    """
    claim = dspy.InputField(desc="The claim to analyze")
    queries: list[str] = dspy.OutputField(desc="Wikipedia-style search queries — proper-noun names — for entities the claim refers to")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25
        self.hop2_queries = 10
        self.hop3_queries = 7
        self.hop4_queries = 6
        self.hop1_keep = 12
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_entities = dspy.Predict(ClaimEntities)
        self.expand_queries = dspy.ChainOfThought(QueryExpansion)
        self.analyze_claim = dspy.ChainOfThought(ClaimAnalysis)

    @staticmethod
    def _title(doc):
        return doc.split(" | ", 1)[0].strip().lower()

    @staticmethod
    def _round_robin(lists):
        seen = set()
        out = []
        max_len = max((len(lst) for lst in lists), default=0)
        for rank in range(max_len):
            for lst in lists:
                if rank < len(lst):
                    doc = lst[rank]
                    t = HoverMultiHop._title(doc)
                    if t not in seen:
                        seen.add(t)
                        out.append(doc)
        return out

    def _retrieve_many(self, queries):
        return [(q, self.retrieve_k(q).passages) for q in queries]

    def _context_docs(self, docs, n=45, max_snippet=1200):
        seen = set()
        uniq = []
        for d in docs:
            t = self._title(d)
            if t not in seen:
                seen.add(t)
                uniq.append(d)
        lines = []
        for d in uniq[:n]:
            parts = d.split(" | ", 1)
            title = parts[0]
            snippet = parts[1][:max_snippet].replace("\n", " ") if len(parts) > 1 else ""
            lines.append(f"{title} | {snippet}")
        return "\n".join(lines)

    def _build_queries(self, raw, seen_queries, limit):
        out = []
        for q in (raw or []):
            q = (q or "").strip()
            ql = q.lower()
            if q and ql not in seen_queries:
                seen_queries.add(ql)
                out.append(q)
            if len(out) >= limit:
                break
        return out

    def _safe_entities(self, claim):
        try:
            return getattr(self.extract_entities(claim=claim), "entities", None) or []
        except Exception:
            return []

    def _safe_expand(self, claim, context):
        try:
            return getattr(self.expand_queries(claim=claim, retrieved_docs=context), "queries", None) or []
        except Exception:
            return []

    def _safe_analyze(self, claim):
        try:
            return getattr(self.analyze_claim(claim=claim), "queries", None) or []
        except Exception:
            return []

    def _select(self, hop1, query_lists):
        """Dilution-robust selection. Keep each query's EXACT-title match if one was retrieved
        (a doc titled exactly the query is unambiguously the targeted article and must outrank
        near-miss disambiguations, e.g. query "Boy Hits Car" keeps the band "Boy Hits Car", not
        "Boy Hits Car (album)"; query "Ross Case" keeps the tennis player, not "In re Ross"); if
        no exact-title doc exists, prefer a disambiguated "<query> (<specifier>)" article (not a
        disambiguation page), else the query's top hit. These per-query headlines are protected
        FIRST so that, when there are many queries, they are not truncated by the 21 cap. Then
        hop1's top docs (claim-named golds land here) are reserved, and finally round-robin fills
        for breadth. This keeps claim-named and directly-targeted golds regardless of how many
        query-lists are used, so extra queries can't dilute them out."""
        seen = set()
        keep = []

        def add(doc):
            t = self._title(doc)
            if t not in seen:
                seen.add(t)
                keep.append(doc)

        def headline(q, lst):
            if not lst:
                return None
            ql = q.strip().lower()
            # 1. An exact-title doc is unambiguously the targeted article.
            for doc in lst:
                if self._title(doc) == ql:
                    return doc
            # 2. A disambiguated article titled "<query> (<specifier>)" (but NOT a
            #    disambiguation page) is the targeted article even when it is not rank 0.
            #    e.g. query "Franz Ferdinand" -> keep "Franz Ferdinand (band)" over the
            #    rank-0 "Archduke Franz Ferdinand of Austria"; query "Boy Hits Car" ->
            #    prefer "Boy Hits Car (band)" over an unrelated rank-0 hit. Take the
            #    earliest such title so a more-notable match (ranked higher) wins.
            for doc in lst:
                t = self._title(doc)
                if t.startswith(ql + " (") and "disambig" not in t:
                    return doc
            # 3. Fall back to the query's top hit.
            return lst[0]

        # Protect each query's headline FIRST — these are the highest-precision
        # targeted articles (exact / disambiguated title matches). When there are
        # many queries, hop1[:hop1_keep] + every headline can exceed 21; adding
        # headlines first guarantees no targeted gold is truncated by the final 21
        # cap. For <= ~9 queries the protected SET is unchanged (only the order
        # is), so this is a no-op there and strictly helps the many-query case.
        for q, lst in query_lists:
            h = headline(q, lst)
            if h is not None:
                add(h)
        # Reserve hop1 top results, but never so many that headlines get pushed
        # past the 21 cap.
        for d in hop1[: min(self.hop1_keep, MAX_RETRIEVED_DOCS - len(keep))]:
            add(d)
        lists = [hop1] + [lst for _, lst in query_lists]
        for d in self._round_robin(lists):
            if len(keep) >= MAX_RETRIEVED_DOCS:
                break
            add(d)
        return keep[:MAX_RETRIEVED_DOCS]

    def forward(self, claim):
        seen_queries = {claim.strip().lower()}
        hop1 = self.retrieve_k(claim).passages

        all_docs = list(hop1)
        other_lists = []

        # Claim-named entities that are NOT already in hop1's top results are queried directly so
        # they land at rank 1. Entities already in hop1's top results are kept by _select, so we
        # skip them to avoid adding redundant query-lists.
        hop1_top_titles = {self._title(d) for d in hop1[: self.hop1_keep]}
        claim_ents = [e for e in self._safe_entities(claim)
                      if self._title(e) not in hop1_top_titles]

        # Hop 2: claim-named entities first, then claim analysis queries (LM knowledge to infer
        # what the claim refers to), then snippet-mined expansion queries.
        context = self._context_docs(all_docs)
        hop2_q = self._build_queries(
            claim_ents + self._safe_analyze(claim) + self._safe_expand(claim, context),
            seen_queries, self.hop2_queries
        )
        if hop2_q:
            hop2_lists = self._retrieve_many(hop2_q)
            other_lists.extend(hop2_lists)
            for _, lst in hop2_lists:
                all_docs.extend(lst)

        # Hop 3: expansion that mines hop 2's newly retrieved snippets for further entities
        # (e.g. a song article naming its video director, an album article naming its band).
        context = self._context_docs(all_docs)
        hop3_q = self._build_queries(
            self._safe_expand(claim, context), seen_queries, self.hop3_queries
        )
        if hop3_q:
            hop3_lists = self._retrieve_many(hop3_q)
            other_lists.extend(hop3_lists)
            for _, lst in hop3_lists:
                all_docs.extend(lst)

        # Hop 4: final expansion pass to catch deeply-nested entities
        context = self._context_docs(all_docs)
        hop4_q = self._build_queries(
            self._safe_expand(claim, context), seen_queries, self.hop4_queries
        )
        if hop4_q:
            hop4_lists = self._retrieve_many(hop4_q)
            other_lists.extend(hop4_lists)
            for _, lst in hop4_lists:
                all_docs.extend(lst)

        final = self._select(hop1, other_lists)
        return dspy.Prediction(retrieved_docs=final)
