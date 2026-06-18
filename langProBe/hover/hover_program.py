import dspy
import unicodedata
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyNextTarget(dspy.Signature):
    """You are retrieving Wikipedia articles to support verification of a multi-hop factual claim.

    Given the claim and Wikipedia passages already retrieved, identify the SINGLE most important
    Wikipedia article still needed to verify the claim.

    Steps:
    1. List ALL named entities explicitly mentioned in the claim (people, places, organizations,
       works, songs, films, awards, titles, routes/road names, botanical genera, named scientific
       theories and concepts, etc.). CRITICAL nuances:
       - If the claim references "the person who wrote/directed/performed/created X", the PERSON
         themselves is a required entity (not just X's article). E.g., if the claim says
         "fronted by [person]", that person's own article is needed; if the claim says
         "the director of [film]", that director's own article is needed.
       - If the claim references a specific season, episode, or event, use the EXACT Wikipedia
         article title (e.g., "2004-05 Memphis Grizzlies season" not just "Memphis Grizzlies";
         "World Without Love" as a song article, not just "Peter and Gordon").
       - Include ALL named entities, even those used only as comparison subjects or secondary
         referents (e.g., "more scope than Robert E. Howard" → Robert E. Howard is required;
         "born before X" → X is required; "partner of Y" → Y is required).
       - Include named routes and road designations explicitly named in the claim (e.g.,
         "New Hampshire Route 124" is a named route — include it as a required entity).
       - List ONLY proper nouns that name specific Wikipedia articles. Do NOT list descriptive
         claim phrases that describe concepts but are not Wikipedia titles. For example, "the
         feather of truth", "the lake of fire", "the weighing mechanism", "a Chinese film
         studio" are DESCRIPTIONS, not Wikipedia article titles — skip them in Step 1 and
         resolve them via Step 4 after retrieving related articles instead.
       SELF-CHECK: Before moving to Step 2, confirm you have listed EVERY proper-noun entity
       that appears verbatim in the claim text, including comparison subjects and date/person
       anchors (e.g., "more scope than Robert E. Howard" → Robert E. Howard is required;
       "born before Robert Jordan" → Robert Jordan is required).
    2. For EACH named entity from step 1, check the retrieved_passages for a passage whose
       ARTICLE TITLE (the text before the " | " separator) matches that entity's name.
       IMPORTANT: An entity is covered ONLY if its own dedicated article title appears —
       a mere mention of the entity inside another article's text does NOT count as covered.
       Concrete example: if you retrieve "Person A | ...text mentioning Person B...", that does
       NOT cover Person B. You still need a passage starting with "Person B | ..." to cover
       Person B. Always check the TITLE before " | ", never the body text, for coverage.
       A disambiguation page (title containing "disambiguation") does NOT count as the article.
       Also treat any entity in fruitless_queries as covered (it was searched and is either not
       in Wikipedia or was already fully retrieved).
       A sub-page article (whose title contains "bibliography", "filmography", "discography",
       "early life", or "health" as a separate word) does NOT count as the main article for
       that entity. Example: "Robert E. Howard bibliography | ..." does NOT cover "Robert E.
       Howard" — you still need "Robert E. Howard | ..." for that. NOTE: ANY parenthetical
       qualifier makes an article the entity's OWN dedicated Wikipedia article — it is NOT
       a sub-page. Examples: "Dave Evans (singer) | ..." DOES cover "Dave Evans"; "F.E.A.R.
       (video game) | ..." DOES cover "F.E.A.R."; "The Secret Agent (TV series) | ..." DOES
       cover "The Secret Agent"; "Moonrunners (film) | ..." DOES cover "Moonrunners". These
       qualifiers simply DISAMBIGUATE the article — they ARE the entity's article, not sub-
       pages. The ONLY exceptions that do NOT cover X are: "X bibliography", "X filmography",
       "X discography", "X early life", "X health" — these are Wikipedia sub-pages. Similarly,
       a name-variant article (e.g., "Pierre Womé | ..." for "Pierre Nlend Womé") counts as
       covered — do not loop on minor name-form differences. Also, a FULL LEGAL NAME form that
       adds middle names or formal names counts as covered by the common-name article already
       retrieved: e.g., if "Boris Becker | ..." is already retrieved, then "Boris Franz Becker"
       is covered — do not re-query the same person under their longer formal name.
    3. ONLY check the entities you explicitly listed in Step 1. Output the FIRST one whose own
       article title is NOT yet retrieved AND is NOT in fruitless_queries. Do NOT introduce any
       new entity name at this stage — if all Step 1 entities are already covered or fruitless,
       go directly to Step 4. Do NOT query descriptive claim phrases excluded from Step 1.
       When querying a WORK whose title might match multiple Wikipedia articles (a film vs. TV
       series, a video game vs. film, a song vs. its composer), include the Wikipedia type
       qualifier in parentheses: e.g., "The Secret Agent (TV series)", "Moonrunners (film)",
       "F.E.A.R. (video game)", "Stranger in Paradise (song)". Use context from the claim and
       retrieved passages to determine the right qualifier.
       CRITICAL TYPE QUALIFIER RULE: When the claim EXPLICITLY states the media type of a work
       (e.g., "a video game", "the film", "a TV series", "the song"), you MUST include that
       type as a Wikipedia qualifier in your query — NEVER use the bare title alone. Example:
       if the claim says "a 2017 video game" and the title is "F.E.A.R.", query "F.E.A.R.
       (video game)" — NOT bare "F.E.A.R." The qualifier is required to distinguish the
       specific article from disambiguation pages or series overview articles.
    4. If ALL named entities in the claim already have their own article title retrieved or are
       in fruitless_queries, scan the body text of EACH retrieved passage for the most important
       named entity not yet retrieved as its own standalone article. Look for:
       - A NAMED SONG, TV SHOW, FILM, OR MUSICAL WORK specifically mentioned in a retrieved
         article as the direct subject of the claim relationship — e.g., the specific song
         title ("A World Without Love") in a band's article as their biggest hit; the specific
         show name ("Punchlines") in a host's or performer's article as the show they hosted
         or appeared on; the specific musical adaptation ("Stranger in Paradise") in a
         composer's article as a work derived from their composition
       - A BOTANICAL GENUS, named scientific theory, or primary research subject discussed in
         a scientist's or researcher's article as their main area of study — e.g., "Crepis"
         in a botanist's article describing their primary research genus (prefer this over
         co-authors or collaborators from the same article)
       - The partner, opponent, or co-participant of an athlete/performer named in a retrieved
         tournament or event article — e.g., if the claim asks about "the partner of X in event Y"
         and the event article lists pairs or participants, output the participant who is directly
         listed as PAIRED WITH the claim's subject X (not the tournament winner or most famous
         participant in the event)
       - The film, TV show, or production in which a person performed stunt work or appeared
       - The company that produced a film, the co-winner of an award, the co-author of a work
       - The broader topic article (the religion, county, or country) that sub-articles describe
         (e.g., if retrieved articles discuss Egyptian deities, the "Ancient Egyptian religion"
         article; if retrieved articles discuss Cork schools, the "County Cork" article)
       - A DEFUNCT, FORMER, or CEASED entity (airline, company, team, organization) mentioned
         by name in a retrieved article, when the claim describes that entity by its properties
         (e.g., "the airline that ceased operations in 2015" or "the former team based in X").
         If a retrieved article (such as an airport, venue, or parent company article) explicitly
         names a defunct entity matching the claim's description, query that defunct entity's
         own Wikipedia article directly. Example: if a retrieved airport article names a defunct
         airline that ceased operations in a specific year, and the claim mentions such an
         airline, query "Air Lituanica" (or the specific defunct airline name) directly.
       - A specific individual described in the claim by their involvement in MULTIPLE named
         organizations or groups (e.g., "former member of Band A, Band B, and Band C", or
         "former bassist/vocalist of X, Y, and Z") — when articles for some of those
         organizations are already retrieved, scan each article for the person listed as a
         member across ALL the named organizations; that person is the implied entity whose own
         article has not yet been retrieved
       - The PARENT COMPANY, MANUFACTURER, or OWNING ORGANIZATION explicitly named in a
         retrieved article as the direct owner or parent of a subsidiary, brand, or product —
         e.g., if a retrieved candy-brand or product article says "wholly owned by [Parent Corp]"
         or "a subsidiary of [Parent Corp]", and the claim asks who MANUFACTURES or OWNS that
         brand, query the parent corporation's Wikipedia article (e.g., "Mars, Incorporated" if
         a product article says it is wholly owned by Mars). ORDERING RULE: Apply this bullet
         ONLY after the product/brand article itself is already in retrieved_passages. When the
         product's own article is NOT yet retrieved, query the PRODUCT FIRST (e.g., "Skittles
         (confectionery)" before "Mars, Incorporated") — reserve the parent company query for
         a later hop after the product article is confirmed retrieved.
       - The specific TOWN, VILLAGE, or DISTRICT that a retrieved station, airport, building,
         or venue is explicitly described as being LOCATED IN, when the claim asks about that
         location's town or the town itself is a required article — e.g., if a retrieved
         railway station article says "located in [Town name]", query that town's Wikipedia
         article directly (e.g., "Afonwen" if Caerwys railway station says it was in Afonwen).
       - The specific VENUE, NIGHTCLUB, ARENA, THEATER, or BUILDING named in a retrieved event
         or festival article as the location where the event takes place, when the claim implies
         that venue's own Wikipedia article is needed — e.g., if a festival article names a
         specific nightclub (e.g., "Flex") as its venue, query that venue's Wikipedia article.
       - The COMPOSER, PLAYWRIGHT, SONGWRITER, or PRIMARY CREATOR of a specific artistic
         work named in a retrieved article about a performer, musician, or artist — when a
         retrieved article about an artist or performer credits a named composer or writer
         for a specific song, musical, or work directly associated with the claim, query
         that creator's own Wikipedia article (e.g., if a performer's article names the
         composer of the specific musical they starred in, and the claim implies a creative
         relationship like "composed by X" or "written by Y", output that composer/writer
         as your query)
       - A COMPARISON or RANKING entity explicitly named in a retrieved article as the
         reference point ranked above or below the article's subject — e.g., if a retrieved
         airport article says "the busiest airport in the UK after Heathrow Airport" or "the
         second-busiest after [Airport X]", query that comparison entity (Heathrow Airport or
         Airport X) directly. Apply this when the claim involves a ranked comparison and the
         specific comparison entity is explicitly named in a retrieved article body.
       PRIORITY: Choose the entity that DIRECTLY satisfies the claim's core relationship (the
       song referenced, the show hosted, the genus studied, the adaptation cited) — NOT peripheral
       mentions like co-authors, technical subcomponents, or historical peoples mentioned only
       as descriptors in a title (e.g., "a dance named after the Cumans" → querying the Cumans
       article is wrong; the required article is the dance or its musical adaptation).
       ANTI-FAME BIAS: When a retrieved article lists multiple people (cast members, participants,
       collaborators), do NOT default to the most famous or prominent name. Instead, identify the
       entity that fills the SPECIFIC ROLE described by the claim — e.g., the person listed as
       X's direct partner (not the tournament winner), the lead star of the SPECIFIC film version
       (not the most famous actor in the list), the cast member who appeared in the SPECIFIC
       other work referenced by the claim.
       Output the most important such implied entity not yet retrieved.
    5. NEVER search for any query listed in previous_queries — those have already been searched,
       and since retrieval is deterministic, repeating a query CANNOT retrieve new documents.
       If step 3 or step 4 would lead you to repeat a previous query, you MUST instead look for
       a DIFFERENT uncovered entity in the claim or retrieved text.

    Output ONLY a concise Wikipedia article title or entity name — nothing else.
    Good examples: "Pablo Escobar", "Apple Inc.", "Gene Kelly", "Sheldon Lee Glashow",
                   "2004-05 Memphis Grizzlies season", "World Without Love", "Warren Fu"
    Bad examples: "Who starred in Narcos?", "Was Steven Weinberg a professor?"
    Do NOT output a question. Do NOT output a sentence. Output a Wikipedia title or entity name only.
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(
        desc="Wikipedia passages already retrieved (format: 'ArticleTitle | text excerpt...'). "
             "An entity is covered ONLY if its article TITLE (before ' | ') appears here."
    )
    previous_queries: str = dspy.InputField(
        desc="Comma-separated list of queries already searched in prior hops. Do NOT repeat ANY of these — "
             "retrieval is deterministic so repeating a query can NEVER retrieve new documents.",
        default="None"
    )
    fruitless_queries: str = dspy.InputField(
        desc="Comma-separated queries that returned ZERO new unique documents after deduplication. "
             "These entities are either NOT in Wikipedia or were ALREADY FULLY RETRIEVED by earlier hops. "
             "Treat any entity in this list as 'covered' when deciding whether all Step 1 entities "
             "are covered. If Step 3 would lead you to query a fruitless entity, skip it and try the "
             "next uncovered Step 1 entity, or proceed to Step 4 if all Step 1 entities are covered/fruitless.",
        default="None"
    )
    query: str = dspy.OutputField(
        desc="A single Wikipedia article title or entity name to search for next — "
             "NOT a question, NOT a sentence"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    NONE_PATTERNS = frozenset({
        "none", "n/a", "na", "null", "nil", "unknown", "n/a.", "none.",
        "not applicable", "no query", "no entity", "no result", ""
    })

    def __init__(self):
        super().__init__()
        # Asymmetric-k: hop1 gets 6 slots, hops 2-4 get 5 each (6+5+5+5=21, zero eviction),
        # hop5 expands to k=12 to catch articles at ranks 8-12.
        self.retrieve_hop1 = dspy.Retrieve(k=6)
        self.retrieve_k = dspy.Retrieve(k=5)
        self.retrieve_hop5 = dspy.Retrieve(k=12)
        self.identify_hop2_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop3_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop4_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop5_target = dspy.ChainOfThought(IdentifyNextTarget)

    @staticmethod
    def _normalize_query(q: str) -> str:
        """Normalize a query string for duplicate detection."""
        q = q.strip().lower()
        # Normalize all dash/hyphen variants to spaces so that e.g.
        # "Conroe-North Houston Regional Airport" and
        # "Conroe North Houston Regional Airport" are treated as duplicates.
        q = q.replace('–', ' ').replace('—', ' ').replace('-', ' ')
        q = ' '.join(q.split())  # collapse multiple spaces
        q = unicodedata.normalize('NFC', q)
        return q

    @staticmethod
    def _get_query_with_retry(predictor, claim, retrieved_passages, previous_queries_str, fruitless_queries_str="None"):
        """Call predictor; if query is a placeholder or duplicates a prior query, retry once with an explicit warning."""
        result = predictor(
            claim=claim,
            retrieved_passages=retrieved_passages,
            previous_queries=previous_queries_str,
            fruitless_queries=fruitless_queries_str,
        )
        query = result.query.strip()

        prev_set = {
            HoverMultiHop._normalize_query(q)
            for q in previous_queries_str.split(",")
            if q.strip() and q.strip() not in ("None", "")
        }
        norm_query = HoverMultiHop._normalize_query(query)

        is_none = norm_query in HoverMultiHop.NONE_PATTERNS
        is_duplicate = (not is_none) and (norm_query in prev_set)

        if is_none or is_duplicate:
            if is_none:
                warning = f"[CRITICAL: '{query}' is not a valid Wikipedia article title — you MUST output a specific named entity]"
            else:
                warning = f"[CRITICAL: '{query}' was already searched — you MUST choose a DIFFERENT uncovered entity]"
            augmented_prev = previous_queries_str + f", {warning}"
            result = predictor(
                claim=claim,
                retrieved_passages=retrieved_passages,
                previous_queries=augmented_prev,
                fruitless_queries=fruitless_queries_str,
            )
            query = result.query.strip()
        return query

    def forward(self, claim):
        seen_titles = set()
        all_previous_queries = []
        all_fruitless_queries = []

        def get_new_unique(docs, query=None):
            """Return only docs with titles not yet seen; track query in all_previous_queries."""
            new = []
            for doc in docs:
                title = doc.split(" | ")[0].strip().lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    new.append(doc)
            if query is not None:
                all_previous_queries.append(query)
                if len(new) == 0:
                    all_fruitless_queries.append(query)
            return new

        def prev_queries_str():
            return ", ".join(all_previous_queries) if all_previous_queries else "None"

        def fruitless_str():
            return ", ".join(all_fruitless_queries) if all_fruitless_queries else "None"

        # HOP 1: Direct retrieval on raw claim with k=6 (exact slot count for hop1 in round-robin)
        hop1_new = get_new_unique(self.retrieve_hop1(claim).passages)

        # HOP 2: context uses top-6 from hop1.
        # With asymmetric-k: hop1=6, hops2-4=5 each (6+5+5+5=21 exactly, zero eviction).
        # Excluding rank-7+ docs from LM's coverage check prevents false-coverage from evicted docs.
        context2 = "\n---\n".join(hop1_new[:6]) if hop1_new else "No passages retrieved yet."
        hop2_query = self._get_query_with_retry(
            self.identify_hop2_target, claim, context2, prev_queries_str(), fruitless_str()
        )
        hop2_new = get_new_unique(self.retrieve_k(hop2_query).passages, hop2_query)

        # HOP 3: context uses top-6 from hop1 + top-5 from hop2 (guaranteed round-robin slots)
        early_docs_ctx = hop1_new[:6] + hop2_new[:5]
        context3 = "\n---\n".join(early_docs_ctx) if early_docs_ctx else "No passages retrieved yet."
        hop3_query = self._get_query_with_retry(
            self.identify_hop3_target, claim, context3, prev_queries_str(), fruitless_str()
        )
        hop3_new = get_new_unique(self.retrieve_k(hop3_query).passages, hop3_query)

        # HOP 4: Final targeted sweep — context uses top-6/5/5 from hops 1/2/3
        early_docs_ctx = hop1_new[:6] + hop2_new[:5] + hop3_new[:5]
        context4 = "\n---\n".join(early_docs_ctx) if early_docs_ctx else "No passages retrieved yet."
        hop4_query = self._get_query_with_retry(
            self.identify_hop4_target, claim, context4, prev_queries_str(), fruitless_str()
        )
        hop4_new = get_new_unique(self.retrieve_k(hop4_query).passages, hop4_query)

        # HOP 5 (conditional): Only execute if IdentifyNextTarget identifies a genuinely new entity.
        # Context shows all docs retrieved so far (generous view for best coverage decision).
        # If hop5 fires, we use 5-way round-robin; if not, preserve 4-way round-robin (no regression).
        # hop5 uses k=12 to catch articles at ranks 8-12.
        context5_docs = hop1_new[:6] + hop2_new[:4] + hop3_new[:4] + hop4_new[:4]
        context5 = "\n---\n".join(context5_docs) if context5_docs else "No passages retrieved yet."

        # Compute set of all normalized queries issued so far (used to determine if hop5 is genuinely new)
        all_normalized_queries = {self._normalize_query(q) for q in all_previous_queries}

        # Use retry mechanism to get the best hop5 query candidate
        hop5_query = self._get_query_with_retry(
            self.identify_hop5_target, claim, context5, prev_queries_str(), fruitless_str()
        )
        hop5_query_norm = self._normalize_query(hop5_query)

        # Seen-titles guard for hop5: if the LM proposed an entity already retrieved, retry once
        if hop5_query_norm in seen_titles:
            augmented_prev5 = (
                prev_queries_str()
                + f", [CRITICAL: '{hop5_query}' is already a retrieved Wikipedia article"
                  f" — query a DIFFERENT uncovered entity not yet in the retrieved passages]"
            )
            retry5_result = self.identify_hop5_target(
                claim=claim,
                retrieved_passages=context5,
                previous_queries=augmented_prev5,
                fruitless_queries=fruitless_str(),
            )
            new5_query = retry5_result.query.strip()
            new5_norm = self._normalize_query(new5_query)
            if new5_norm and new5_norm not in seen_titles:
                hop5_query = new5_query
                hop5_query_norm = new5_norm

        hop5_new = []
        if hop5_query_norm and hop5_query_norm not in all_normalized_queries:
            # Genuinely new entity identified — execute hop 5 retrieval with expanded k=12
            hop5_new = get_new_unique(self.retrieve_hop5(hop5_query).passages, hop5_query)

        if hop5_new:
            # Priority interleaving: hop1 gets its full 6 slots first (positions 1-6),
            # then hops 2-5 do a 4-way round-robin for positions 7-21.
            # Slot allocation: hop1→6, hop2→4 (pos 7,11,15,19), hop3→4 (pos 8,12,16,20),
            #                  hop4→4 (pos 9,13,17,21), hop5→3 (pos 10,14,18).
            # This preserves hop1's 6-slot allocation from the 4-hop design, eliminating
            # the false-coverage bug where hop1 rank-6 docs appear in context but are
            # excluded from the final output.
            interleaved = list(hop1_new[:6])
            secondary_hops = [hop2_new, hop3_new, hop4_new, hop5_new]
            max_secondary = max((len(h) for h in secondary_hops), default=0)
            for i in range(max_secondary):
                for hop_docs in secondary_hops:
                    if i < len(hop_docs):
                        interleaved.append(hop_docs[i])
        else:
            # No hop5 results — use standard 4-hop equal round-robin (unchanged from baseline).
            all_hop_docs = [hop1_new, hop2_new, hop3_new, hop4_new]
            interleaved = []
            max_len = max((len(h) for h in all_hop_docs), default=0)
            for i in range(max_len):
                for hop_docs in all_hop_docs:
                    if i < len(hop_docs):
                        interleaved.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=interleaved[:21])
