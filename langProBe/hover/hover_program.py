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
      - A work's snippet naming its author/director/creator -> that person's article
        (e.g. "The Broken Tower" by Hart Crane -> query "Hart Crane").

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
    - The `issued_queries` field lists EVERY query already issued (lowercased). Do NOT repeat any
      of them — they are filtered out, so repeating them wastes the query budget. Instead mine NEW
      proper nouns from snippets that are NOT in issued_queries and NOT already a retrieved TITLE.
      Scan every snippet (especially freshly retrieved ones) for proper nouns the claim depends on
      and output a query for EACH new one. A single snippet often lists MANY such names (full cast,
      every band member, every partner) — output ALL of them, not just the first.
    - If the claim specifies a MEDIUM/TYPE for an entity a snippet names (film, song, album, comic
      book, video game, TV series, novel, band, play, musical), append the Wikipedia type
      parenthetical so ColBERT retrieves the typed article instead of a different-type article
      sharing the bare name:
        - claim says "film" + snippet names "The Grapes of Wrath" -> "The Grapes of Wrath (film)"
        - claim says "comic book" + the publication article is meant -> "Archie (comic book)"
        - claim says "song" + snippet names "Figure It Out" -> "Figure It Out (song)"
      Do this EVEN IF the bare name already has a retrieved article of a different type (e.g. the
      novel) — the typed article is a DISTINCT supporting document.
    - Cover DISTINCT missing entities; queries must be non-redundant.
    - NEVER output a role/description ("director of X", "host of X", "remake of X", "winner of X").
      Always output a PROPER-NOUN NAME. If you cannot name it from a snippet or knowledge, SKIP it
      rather than emit a description.
    - When unsure whether an entity is covered, output a query rather than none — a missed document
      zeros the entire score.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    retrieved_docs = dspy.InputField(desc="Documents retrieved so far as '<title> | <snippet>', one per line")
    issued_queries = dspy.InputField(desc="Every search query already issued (lowercased); do NOT repeat any — mine NEW proper nouns from snippets instead")
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
      X" -> output BOTH the character article ("Archie Andrews") AND the comic book series article
      ("Archie (comic book)"); an album -> also the artist's article; a song -> also the album.
    - ENUMERATE ALL members of a referenced group/movement/category/trend. When the claim says
      "another writer associated with the Nouveau Roman trend along with Alain Robbe-Grillet",
      output EVERY member of that group you can name (Claude Simon, Nathalie Sarraute, Marguerite
      Duras, Michel Butor, Robert Pinget, ...), not just one or the group article — the gold
      document is whichever member matches and you cannot know in advance which. Likewise for "the
      other X", "a member of Y", "along with Z": list ALL plausible candidates as separate queries.
    - Infer COMPARATIVE / CONTRASTIVE entities the claim implies but does not name. When the claim
      says "the busiest airport outside of London", the busiest INSIDE London (Heathrow Airport) is
      an implied comparator and a supporting document — output it. When the claim says "not another
      writer", that other writer is implied — name them if you can. A superlative/comparison almost
      always implies the entity it is measured against; output that implied entity's article too.
    - When unsure, output a query rather than nothing — missing documents cost the entire score.
    """
    claim = dspy.InputField(desc="The claim to analyze")
    queries: list[str] = dspy.OutputField(desc="Wikipedia-style search queries — proper-noun names — for entities the claim refers to")


class DisambiguateEntity(dspy.Signature):
    """A bare Wikipedia entity name is AMBIGUOUS: querying it retrieved several disambiguated
    articles of DIFFERENT types (films, games, other topics) but NOT the intended article.
    Determine the intended article's TYPE from the claim and the retrieved snippets, then output
    the query WITH the correct Wikipedia type parenthetical so ColBERT retrieves that exact article.

    This is PURE DOCUMENT RETRIEVAL — do NOT verify, fact-check, or judge the claim.

    How to find the type:
    - Scan the context snippets for each ambiguous name; the surrounding words usually state its
      type:
        "the glam-style metal band It's Alive"  -> type is band  -> "It's Alive (band)"
        "the 1974 film It's Alive"              -> type is film  -> "It's Alive (film)"
        "Crank, the 1994 album by ..."          -> type is album -> "Crank (album)"
    - Common parentheticals: (band), (film), (song), (album), (TV series), (video game),
      (novel), (book), (magazine), (play), (miniseries), (franchise).
    - If the type is genuinely uncertain between two plausible types, output BOTH typed variants
      as separate queries. If you cannot determine the type at all, output an empty list.

    Rules:
    - Each query is "<bare name> (<type>)" exactly, matching Wikipedia's disambiguation convention.
    - Do NOT output the bare name again; do NOT output descriptions, roles, or full sentences.
    - Output one typed query per ambiguous name (or two if the type is uncertain).
    """
    claim = dspy.InputField(desc="The claim")
    ambiguous_names = dspy.InputField(desc="Each ambiguous bare name followed by the disambiguated titles retrieved for it")
    context_snippets = dspy.InputField(desc="Retrieved snippets that may state each entity's type, '<title> | <snippet>' one per line")
    queries: list[str] = dspy.OutputField(desc='Type-disambiguated queries like "It\'s Alive (band)"; empty list if type unknown')


class LinkingEntity(dspy.Signature):
    """The claim names several entities that share an UNNAMED work or relation — a movie, film,
    musical, comic book / comic strip, album, song, TV show, band, book, video game, a city a
    person founded, a person's spouse or relative, a director/creator of a work, etc. The claim
    refers to this unnamed entity only RELATIONALLY ("the movie X starred in", "the comic book
    character", "the recording in the discography", "the city X founded", "the spouse of X",
    "the band behind the album", "the director of X").

    Identify the SINGLE Wikipedia article that is this unnamed LINKING entity — the work or
    person that the named entities in the claim SHARE. Use world knowledge.

    This is PURE DOCUMENT RETRIEVAL — do NOT verify whether the claim is true. Just NAME the
    linking entity.

    ## The intersection is the key signal
    Name the work/relation that ALL (or the relevant pair) of the named entities share, NOT a
    work that only ONE of them is famous for. If a film starring actor A also features actors B
    and C, that film is the linking entity — even if A is far more famous for other films. Walk
    each named person's filmography / discography / credits until you find the ONE title they
    all share; that title is the linking entity.
      - "actor A and actor B starred in the same film" -> the single film both A and B appear in.
      - "the comic book character that first appeared in magazine M" -> the comic book / comic
        strip series the character headlines, as its OWN article with the type parenthetical, e.g.
        "Archie (comic book)" (separate from the character article "Archie Andrews").
      - "X used his inheritance to fund the founding of the city" -> the city, AND possibly X's
        SPOUSE (a founder's spouse is often a separate supporting article).
      - "the recording in Y's discography whose choreographer was born in 1912" -> the specific
        cast album / musical / recording in Y's discography.

    ## Also infer unnamed immediate family / co-creators
    When a claim about a person's founding, marriage, or inheritance implies a SPOUSE,
    co-founder, or close relative, name that relative too (a founder's wife/husband, the
    co-writer of a work, etc., are common separate supporting articles).

    Rules:
    - Output proper-noun Wikipedia titles ONLY, with the disambiguator parenthetical when needed
      (e.g. "Some Film (film)", "Some Musical (musical)", "Some Comic (comic book)"). Never
      output a description, role, or full sentence.
    - If several linking entities are plausible, output EACH as a separate query.
    - If the claim already names every entity it depends on (no unnamed linking entity), output
      an EMPTY list. Do NOT invent unrelated entities.
    """
    claim = dspy.InputField(desc="The claim")
    named_entities = dspy.InputField(desc="Comma-separated entities explicitly named in the claim")
    queries: list[str] = dspy.OutputField(desc="Wikipedia titles of the unnamed linking entities, or an empty list if the claim names everything")


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
        self.disambiguate = dspy.ChainOfThought(DisambiguateEntity)
        self.linking = dspy.ChainOfThought(LinkingEntity)

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

    def _context_docs(self, docs, n=60, max_snippet=1500):
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

    def _safe_expand(self, claim, context, issued_queries):
        try:
            return getattr(self.expand_queries(claim=claim, retrieved_docs=context, issued_queries=issued_queries), "queries", None) or []
        except Exception:
            return []

    def _safe_analyze(self, claim):
        try:
            return getattr(self.analyze_claim(claim=claim), "queries", None) or []
        except Exception:
            return []

    def _safe_linking(self, claim, entities):
        try:
            ents = [e for e in (entities or []) if e]
            r = self.linking(claim=claim, named_entities=", ".join(ents))
            return getattr(r, "queries", None) or []
        except Exception:
            return []

    def _ambiguous(self, query_lists):
        """Find bare queries (no parenthetical) that retrieved >=2 disambiguated variants of the
        form '<query> (<specifier>)' (excluding disambiguation pages) but no exact-title match.
        Such names are shared by multiple Wikipedia articles, so the bare query likely retrieved
        the wrong one; a type-disambiguated query (e.g. 'It's Alive (band)') is needed."""
        amb = []
        for q, lst in query_lists:
            ql = q.strip().lower()
            if "(" in ql or not lst:
                continue
            if any(self._title(d) == ql for d in lst):
                continue
            variants = [d for d in lst
                        if self._title(d).startswith(ql + " (") and "disambig" not in self._title(d)]
            if len(variants) >= 2:
                amb.append((q, variants))
        return amb

    def _safe_disambig(self, claim, ambiguous, all_docs):
        try:
            names_lc = {q.strip().lower() for q, _ in ambiguous}
            focused = []
            seen_t = set()
            for d in all_docs:
                t = self._title(d)
                if t in seen_t:
                    continue
                seen_t.add(t)
                blob = d.lower()
                if any(n in blob for n in names_lc):
                    parts = d.split(" | ", 1)
                    title = parts[0]
                    snippet = parts[1][:1500].replace("\n", " ") if len(parts) > 1 else ""
                    focused.append(f"{title} | {snippet}")
            context = "\n".join(focused[:40])
            block = "\n".join(
                f"NAME: {q}\nVARIANTS: " + "; ".join(self._title(d) for d in variants)
                for q, variants in ambiguous
            )
            r = self.disambiguate(
                claim=claim, ambiguous_names=block, context_snippets=context
            )
            return getattr(r, "queries", None) or []
        except Exception:
            return []

    @staticmethod
    def _ordinal_variants(query):
        """Generate alternate ordinal spellings. Wikipedia article titles use both '3rd'/'3d' and
        '2nd'/'2d'; a claim may use one form while the gold title uses the other."""
        import re
        variants = []
        for num, suffix, alt in [("2", "nd", "d"), ("3", "rd", "d")]:
            pattern = re.compile(re.escape(num + suffix), re.IGNORECASE)
            if pattern.search(query):
                variant = pattern.sub(num + alt, query)
                if variant != query:
                    variants.append(variant)
        return variants

    def _select(self, hop1, query_lists):
        """Dilution-robust selection under the 21-doc cap.

        Each query yields a "headline" — the doc most likely to be that query's targeted gold:
          1. an EXACT-title match (query == title) — unambiguously the targeted article
             (e.g. query "Boy Hits Car" keeps the band, not "Boy Hits Car (album)");
          2. a disambiguated "<query> (<specifier>)" article (not a disambiguation page)
             (e.g. query "Franz Ferdinand" -> "Franz Ferdinand (band)");
          3. else the query's top ColBERT hit (a lower-precision guess).

        Protection order (so extra mining queries can NEVER dilute out a gold):
          a. EXACT headlines — highest precision, protect first.
          b. hop1[:hop1_keep] — retrieve(claim)'s top docs, where claim-named golds land. These
             are NOT any query's headline when the entity was already in hop1 (it is skipped to
             avoid a redundant search), so they must be reserved explicitly BEFORE the many
             lower-precision headlines, or they get squeezed past rank 21.
          c. DISAMBIGUATED headlines — high-precision targeted articles.
          d. Round-robin fill from [hop1] + every query list — this naturally admits the top-hit
             headlines and any deeper golds, so top-hit headlines are NOT pre-protected (they are
             the LM's guesses and protecting them all is what used to dilute hop1 claim-golds)."""
        seen = set()
        keep = []

        def add(doc):
            t = self._title(doc)
            if t not in seen:
                seen.add(t)
                keep.append(doc)

        def headline(q, lst):
            """Return (doc, kind) where kind in {'exact','disambig','top'}, or (None, None)."""
            if not lst:
                return None, None
            ql = q.strip().lower()
            for doc in lst:
                if self._title(doc) == ql:
                    return doc, "exact"
            for doc in lst:
                t = self._title(doc)
                if t.startswith(ql + " (") and "disambig" not in t:
                    return doc, "disambig"
            return lst[0], "top"

        exact_heads, disambig_heads = [], []
        for q, lst in query_lists:
            h, kind = headline(q, lst)
            if h is None:
                continue
            if kind == "exact":
                exact_heads.append(h)
            elif kind == "disambig":
                disambig_heads.append(h)

        for d in exact_heads:
            add(d)
        for d in hop1[: min(self.hop1_keep, MAX_RETRIEVED_DOCS - len(keep))]:
            add(d)
        for d in disambig_heads:
            if len(keep) >= MAX_RETRIEVED_DOCS:
                break
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
        # skip them to avoid adding redundant query-lists (which would crowd the hop2 query budget
        # and displace the snippet-mining/analysis queries that find non-claim-named golds).
        hop1_top_titles = {self._title(d) for d in hop1[: self.hop1_keep]}
        raw_entities = self._safe_entities(claim)
        claim_ents = [e for e in raw_entities
                      if self._title(e) not in hop1_top_titles]

        # Linking-entity inference: a claim often refers to an UNNAMED work/person that the named
        # entities share (the film several actors appeared in, the comic book a character
        # headlines, the spouse of a named founder, the recording in a discography). The general
        # ClaimAnalysis/expansion prompts frequently miss these because the gold is the
        # INTERSECTION of the named entities' credits, not any single person's most-famous work.
        # This focused module names that intersection. It is SELF-GATING: for claims that already
        # name every entity, it returns an empty list, so passing examples incur no extra searches.
        # Run before hop2 so hop2's snippet-mining expansion sees the linking articles' snippets.
        linking_q = self._build_queries(self._safe_linking(claim, raw_entities), seen_queries, 4)
        if linking_q:
            linking_lists = self._retrieve_many(linking_q)
            other_lists.extend(linking_lists)
            for _, lst in linking_lists:
                all_docs.extend(lst)

        # Hop 2: claim-named entities first, then claim analysis queries (LM knowledge to infer
        # what the claim refers to), then snippet-mined expansion queries.
        context = self._context_docs(all_docs)
        hop2_q = self._build_queries(
            claim_ents + self._safe_analyze(claim) + self._safe_expand(claim, context, sorted(seen_queries)),
            seen_queries, self.hop2_queries
        )
        # Add ordinal spelling variants (e.g. "3rd Pursuit Group" -> "3d Pursuit Group") —
        # Wikipedia titles use both forms and ColBERT won't match across the spelling difference.
        hop2_variants = []
        for q in hop2_q:
            for v in self._ordinal_variants(q):
                vl = v.lower()
                if vl not in seen_queries:
                    seen_queries.add(vl)
                    hop2_variants.append(v)
        hop2_q = hop2_q + hop2_variants
        if hop2_q:
            hop2_lists = self._retrieve_many(hop2_q)
            other_lists.extend(hop2_lists)
            for _, lst in hop2_lists:
                all_docs.extend(lst)

        # Disambiguation refinement: a bare entity name (e.g. "It's Alive") may retrieve several
        # wrong-type disambiguated articles (films, games) but not the intended one (the band).
        # Detect such ambiguous queries and issue type-disambiguated variants ("It's Alive (band)")
        # so ColBERT surfaces the targeted article. The new results are MERGED INTO the original
        # bare query's list (rather than added as new query-lists) so the targeted article becomes
        # that query's headline WITHOUT creating an extra headline slot that could displace other
        # golds under the 21-doc cap. Placed before hop3 so later hops can expand on the new article.
        amb = self._ambiguous(other_lists)
        if amb:
            list_for = {q: lst for q, lst in other_lists}
            typed_q = self._build_queries(
                self._safe_disambig(claim, amb, all_docs), seen_queries, 8
            )
            if typed_q:
                import re as _re
                for tq, lst in self._retrieve_many(typed_q):
                    all_docs.extend(lst)
                    bare = _re.sub(r"\s*\([^()]*\)\s*$", "", tq).strip()
                    target = list_for.get(bare)
                    if target is not None:
                        # Prepend in place so the bare query's headline picks up the typed article.
                        target[0:0] = lst

        # Hop 3: expansion that mines hop 2's newly retrieved snippets for further entities
        # (e.g. a song article naming its video director, an album article naming its band).
        context = self._context_docs(all_docs)
        hop3_q = self._build_queries(
            self._safe_expand(claim, context, sorted(seen_queries)), seen_queries, self.hop3_queries
        )
        if hop3_q:
            hop3_lists = self._retrieve_many(hop3_q)
            other_lists.extend(hop3_lists)
            for _, lst in hop3_lists:
                all_docs.extend(lst)

        # Hop 4: final expansion pass to catch deeply-nested entities
        context = self._context_docs(all_docs)
        hop4_q = self._build_queries(
            self._safe_expand(claim, context, sorted(seen_queries)), seen_queries, self.hop4_queries
        )
        if hop4_q:
            hop4_lists = self._retrieve_many(hop4_q)
            other_lists.extend(hop4_lists)
            for _, lst in hop4_lists:
                all_docs.extend(lst)

        final = self._select(hop1, other_lists)
        return dspy.Prediction(retrieved_docs=final)
