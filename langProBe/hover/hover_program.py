import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class SummarizeHop1(dspy.Signature):
    """You are retrieving Wikipedia documents to verify a multi-hop claim.

    Given the claim and the Wikipedia abstract passages retrieved so far, write a
    concise factual summary that:
    1. Extracts the key named entities (people, organizations, places, works,
       events) appearing in the passages and the relationships among them.
    2. Identifies which entities mentioned in the passages (e.g. via phrases
       like "produced by X", "dedicated to Y", "located beside Z", "directed by
       W") still need their OWN Wikipedia article retrieved to verify or refute
       the claim. This is multi-hop retrieval: a passage often names the
       next-hop entity to chase.
    3. Notes any entity that has already been covered vs. one still missing.

    Do not omit named entities that appear in the passages even if they seem
    tangential; the next hop will use them as search targets."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    summary: str = dspy.OutputField(desc="Concise entity-focused summary of passages")


class SummarizeHop2(dspy.Signature):
    """You are retrieving Wikipedia documents to verify a multi-hop claim and
    have already completed two retrieval hops.

    Given the claim, the prior summary, the new passages, and the list of
    Wikipedia article titles ALREADY retrieved, write an updated concise factual
    summary that:
    1. Integrates the new passages with the prior summary, tracking which
       supporting entities have now been found and which are still missing.
       IMPORTANT: an entity is "found" ONLY if its exact Wikipedia article title
       appears in retrieved_titles. An entity merely *mentioned* inside a
       passage is NOT yet retrieved.
    2. Emphasize every named entity mentioned in the passages via a relational
       phrase (e.g. "produced by X", "directed by Y", "dedicated to Z",
       "wife of W", "born in P") whose OWN Wikipedia article is NOT in
       retrieved_titles and is needed to verify or refute the claim.
    3. Explicitly list the candidate entities still missing (not in
       retrieved_titles) that should be searched next.

    Be precise about entity names; they will become search queries."""
    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="Summary from the prior hop")
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hop 1 and hop 2")
    summary: str = dspy.OutputField(desc="Updated entity-focused summary explicitly listing missing entities")


class CreateQueryHop2(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Hop 1 retrieved passages using the raw claim (the prior query) and they
    were summarized. Now generate ONE search query for hop 2 that follows the
    multi-hop chain to an entity still missing.

    DO:
    - Pick a named entity mentioned in the summary (often introduced by hop-1
      passages via a relational phrase such as "produced by X",
      "dedicated to Y", "located beside Z", "directed by W", "wife of W")
      whose OWN Wikipedia article is NOT in retrieved_titles.
    - Use the bare entity name; append a single disambiguator (e.g. "film",
      "band", "actor", "company", "place") only when the name is genuinely
      ambiguous.

    DO NOT:
    - Repeat or restate the prior query (the raw claim) or any phrase that
      would just re-retrieve the same articles.
    - Output "none", "no query", "verification complete", an empty string, or
      any meta-commentary. Always output exactly one concrete entity search
      query.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved in hop 1")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued; do not repeat them")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class CreateQueryHop3(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Two hops of retrieval have been completed. Generate ONE search query for
    hop 3 that follows the multi-hop chain to an entity STILL missing.

    DO:
    - From the summaries, identify a named entity mentioned by the retrieved
      passages via a relational phrase (e.g. "produced by X", "dedicated to Y",
      "directed by W", "wife of W", "located beside Z", "born in P", "starring Z")
      whose OWN Wikipedia article is NOT in retrieved_titles and is needed to
      verify or refute the claim.
    - Prefer a specific PERSON (the actor, director, singer, writer, scientist)
      when one is named in the passages and not yet retrieved — biographical
      articles are the most common missing hop.
    - Target that entity's Wikipedia article. Prefer the bare entity name;
      append a single disambiguator only when the name is genuinely ambiguous.
    - If several entities are still missing, pick a DIFFERENT entity than any
      in prior_queries.

    DO NOT:
    - Output "none", "no query", "no missing entity", "verification complete",
      an empty string, or any meta-commentary. Even if the claim looks
      verified, there is almost always a multi-hop neighbour still missing:
      always output exactly one concrete entity search query for that
      neighbour.
    - Restate the whole claim or repeat a prior query; pick a new specific
      missing entity.
    - Pivot to a derived/connector entity (film festival, song title, TV
      series name, continent) when a more specific PERSON is named in the
      passages and not yet retrieved.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    summary_2: str = dspy.InputField(desc="Updated entity-focused summary after hop 2, listing missing entities")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hop 1 and hop 2")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued; do not repeat them")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class CreateQueryHop4(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    THREE retrieval hops have been completed. This is the FINAL gap-analysis
    hop: produce ONE search query targeting an entity STILL missing after the
    first three hops.

    DO:
    - Re-read the summary; identify a NAMED ENTITY (especially a PERSON — but
      also a place, organization, or work) that the retrieved passages NAME via
      a relational phrase such as "directed by X", "starring Y", "wife of W",
      "born in P", "produced by Z", "dares with Q", "named after R", "dedicated
      to S" whose OWN Wikipedia article is NOT in retrieved_titles and is
      needed to verify or refute the claim.
    - Prefer a specific PERSON when several are missing: biographical articles
      are the most common remaining hop.
    - Use the bare entity name (first + last as it appears in the passage);
      append a single disambiguator (actor, singer, director, film, player,
      place) only if the name is genuinely ambiguous.

    DO NOT:
    - Repeat any prior_queries or restate the claim.
    - Pivot to a derived/connector entity (film festival, song title, TV series
      name, continent, dataset, disambiguation page) when a more specific PERSON
      is named in the passages and not yet retrieved — follow to the person.
    - Output "none", "no query", "no missing entity", "verification complete",
      an empty string, or any meta-commentary. Always output exactly one
      concrete entity search query.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary: str = dspy.InputField(desc="Latest entity-focused summary listing entities still missing")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hops 1-3; do not query these")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued; do not repeat them")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity (prefer a person)")


class HarvestEntityQueries(dspy.Signature):
    """You are helping multi-hop Wikipedia retrieval. Several retrieval hops
    have been completed. Some entities NAMED in the retrieved passages may
    still need their OWN Wikipedia article retrieved.

    Inspect ALL passages and identify up to 3 entity NAMES (people, places,
    organizations, works) that:
    1. Appear literally in the passage text (often via relational phrases
       like "directed by X", "produced by Y", "wife of Z", "born in P",
       "named after R", "dedicated to S").
    2. Do NOT have their own Wikipedia article title in retrieved_titles.
    3. Are plausibly needed to verify or refute the claim.

    Output only the bare entity names as they would appear as Wikipedia article
    titles. If no new entity is named that is not already retrieved, output an
    empty list."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="All Wikipedia passages retrieved across all hops")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved")
    queries: list[str] = dspy.OutputField(desc="Up to 3 bare entity names whose articles are not yet retrieved")


class BridgeDescriptorQuery(dspy.Signature):
    """You are helping multi-hop Wikipedia retrieval. The claim may refer to
    an entity by ROLE, title, location, or descriptor without naming it, and
    NO retrieved passage names it either.

    Identify ONE entity that:
    1. The claim implies by role or descriptor (e.g. "the director of a film
       named in the claim", "the company that produced a product mentioned",
       "the place where someone was born", "the subsidiary of an airline").
    2. Is NOT named in any retrieved passage.
    3. Is NOT in retrieved_titles.

    Output the single bare entity NAME as it would appear as a Wikipedia
    article title. If you cannot identify such an entity, output an empty
    string."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="All Wikipedia passages retrieved across all hops")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved")
    query: str = dspy.OutputField(desc="A single bare entity name to search for, or empty string if none")


class EntityHop1Query(dspy.Signature):
    """You are starting multi-hop Wikipedia retrieval for a claim.

    Generate a search query targeting the MAIN ENTITY of the claim with a type
    disambiguator appended. This helps retrieve the correct Wikipedia article
    when the entity name is ambiguous.

    Examples:
    - A claim about a TV series -> "entity name television series" or "tv series"
    - A claim about a film -> "entity name film"
    - A claim about a song -> "entity name song"
    - A claim about a sports season -> "entity name season"
    - A claim about a board game -> "entity name board game"
    - A claim about a person -> bare "entity name" (no disambiguator needed)

    Output a single concise search query with the entity name and optional type
    disambiguator."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="Entity name with optional type disambiguator appended")


class RerankDocsStage1(dspy.Signature):
    """You are scoring Wikipedia article titles for relevance to a multi-hop
    claim. Each title may be a supporting document needed to verify or refute
    the claim.

    Assign a relevance score from 0 to 10 for each title:
    - 10: directly named or the central entity in the claim
    - 7-9: a person, place, organization, or work directly connected to the
      claim through a multi-hop chain (e.g. the director of a film named in
      the claim, the company that produced a product mentioned)
    - 4-6: plausibly related but tangential
    - 0-3: unrelated

    Return one score per title, in the same order as the input titles."""
    claim: str = dspy.InputField()
    titles: list[str] = dspy.InputField(desc="Wikipedia article titles to score for claim relevance")
    scores: list[float] = dspy.OutputField(desc="Relevance scores 0-10, one per title in order")


class RerankDocsStage2(dspy.Signature):
    """You are re-scoring Wikipedia article titles for relevance to a
    multi-hop claim. This is a second independent evaluation of the same
    titles.

    Assign a relevance score from 0 to 10 for each title:
    - 10: directly named or the central entity in the claim
    - 7-9: a person, place, organization, or work directly connected to the
      claim through a multi-hop chain
    - 4-6: plausibly related but tangential
    - 0-3: unrelated

    Return one score per title, in the same order."""
    claim: str = dspy.InputField()
    titles: list[str] = dspy.InputField(desc="Wikipedia article titles to re-score")
    scores: list[float] = dspy.OutputField(desc="Re-scored relevance 0-10, one per title in order")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15
        self.final_doc_limit = 21
        self.harvest_query_cap = 3
        self.rrf_k_constant = 60
        self.rrf_pool_cap = 45
        self.stage1_keep_margin = 9

        self.create_query_hop2 = dspy.ChainOfThought(CreateQueryHop2)
        self.create_query_hop3 = dspy.ChainOfThought(CreateQueryHop3)
        self.create_query_hop4 = dspy.ChainOfThought(CreateQueryHop4)
        self.create_entity_hop1 = dspy.ChainOfThought(EntityHop1Query)
        self.harvest = dspy.ChainOfThought(HarvestEntityQueries)
        self.bridge = dspy.ChainOfThought(BridgeDescriptorQuery)
        self.rerank_stage1 = dspy.ChainOfThought(RerankDocsStage1)
        self.rerank_stage2 = dspy.ChainOfThought(RerankDocsStage2)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought(SummarizeHop1)
        self.summarize2 = dspy.ChainOfThought(SummarizeHop2)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _titles(passages):
        """Extract the leading Wikipedia title from each 'Title | ...' passage."""
        return [p.split(" | ", 1)[0] for p in passages]

    @staticmethod
    def _is_refusal(q):
        """True if the model emitted a degenerate refusal/no-query."""
        if q is None:
            return True
        cleaned = q.strip().lower().rstrip(".!,;:")
        refusal_tokens = {
            "",
            "none",
            "no query",
            "no missing entity",
            "no missing",
            "no further retrieval required",
            "no further query",
            "no further",
            "no additional query",
            "no additional retrieval required",
            "not required",
            "verification complete",
            "claim verified",
            "n/a",
            "na",
            "no entity",
            "no entity needed",
            "no query needed",
            "no query required",
            "completed",
            "done",
        }
        return cleaned in refusal_tokens

    @staticmethod
    def _is_disambig(title):
        """True if title is a Wikipedia disambiguation navigational page."""
        t = title.lower().strip()
        return (
            t.endswith("(disambiguation)")
            or t.endswith("(disambiguation page)")
            or t == "disambiguation"
        )

    def _safe_query(self, module, fallback, **kwargs):
        """Call a ChainOfThought module and extract its 'query' field,
        falling back to *fallback* on any parse/error."""
        try:
            result = module(**kwargs)
            q = getattr(result, "query", None)
            if q is None or (isinstance(q, str) and not q.strip()):
                return fallback
            return q
        except Exception:
            return fallback

    def _safe_summary(self, module, **kwargs):
        """Call a ChainOfThought summarizer, falling back to '' on error."""
        try:
            result = module(**kwargs)
            return getattr(result, "summary", "")
        except Exception:
            return ""

    def _safe_harvest(self, claim, passages, retrieved_titles):
        """Call the harvest module and return a list of entity-name queries."""
        try:
            result = self.harvest(
                claim=claim,
                passages=passages,
                retrieved_titles=retrieved_titles,
            )
            queries = getattr(result, "queries", None)
            if queries is None:
                return []
            if isinstance(queries, str):
                queries = [q.strip() for q in queries.split(",") if q.strip()]
            return [q for q in queries if q and not self._is_refusal(q)]
        except Exception:
            return []

    def _safe_bridge(self, claim, passages, retrieved_titles):
        """Call the bridge module and return a single entity-name query or None."""
        try:
            result = self.bridge(
                claim=claim,
                passages=passages,
                retrieved_titles=retrieved_titles,
            )
            q = getattr(result, "query", None)
            if q is None or self._is_refusal(q):
                return None
            return q.strip()
        except Exception:
            return None

    def _safe_rerank(self, module, claim, titles):
        """Call a reranker module and return a list of float scores, or None."""
        if not titles:
            return []
        try:
            result = module(claim=claim, titles=titles)
            scores = getattr(result, "scores", None)
            if scores is None:
                return None
            scores = [float(s) for s in scores]
            if len(scores) != len(titles):
                return None
            return scores
        except Exception:
            return None

    def _compute_rrf(self, hop_docs_map):
        """Reciprocal Rank Fusion: for each unique title, sum 1/(k+rank) across
        all retrieval sources. Returns dict title -> RRF score normalised to
        [0, 10] where 10 goes to the highest-RRF title."""
        rrf = {}
        for docs in hop_docs_map.values():
            for rank, passage in enumerate(docs):
                title = passage.split(" | ", 1)[0]
                rrf[title] = rrf.get(title, 0.0) + 1.0 / (self.rrf_k_constant + rank)
        if not rrf:
            return {}
        max_score = max(rrf.values())
        if max_score > 0:
            rrf = {t: (s / max_score) * 10.0 for t, s in rrf.items()}
        return rrf

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, claim):
        hop_docs = {}  # source_name -> list[passage]

        # ---- HOP 1: raw claim retrieval ----
        hop1_docs = self.retrieve_k(claim).passages
        hop_docs["hop1"] = hop1_docs

        # ---- HOP 1b: entity disambiguator query ----
        hop1b_q = self._safe_query(self.create_entity_hop1, fallback="", claim=claim)
        if self._is_refusal(hop1b_q):
            hop1b_docs = []
        else:
            hop1b_docs = self.retrieve_k(hop1b_q.strip()).passages
        hop_docs["hop1b"] = hop1b_docs

        # ---- Summarize hop 1 (include hop1b passages so disambiguator-
        # retrieved entities enter the multi-hop chain) ----
        summary_1 = self._safe_summary(
            self.summarize1, claim=claim, passages=hop1_docs + hop1b_docs
        )

        all_titles = list(dict.fromkeys(self._titles(hop1_docs) + self._titles(hop1b_docs)))

        # ---- HOP 2 ----
        hop2_q = self._safe_query(
            self.create_query_hop2,
            fallback="",
            claim=claim,
            summary_1=summary_1,
            retrieved_titles=all_titles,
            prior_queries=[claim],
        )
        if self._is_refusal(hop2_q):
            hop2_docs = []
        else:
            hop2_q = hop2_q.strip()
            hop2_docs = self.retrieve_k(hop2_q).passages
        hop_docs["hop2"] = hop2_docs
        all_titles = list(dict.fromkeys(all_titles + self._titles(hop2_docs)))

        # ---- Summarize hop 2 ----
        summary_2 = self._safe_summary(
            self.summarize2,
            claim=claim,
            context=summary_1,
            passages=hop2_docs,
            retrieved_titles=all_titles,
        )

        # ---- HOP 3 ----
        prior_q = [claim]
        if not self._is_refusal(hop2_q):
            prior_q.append(hop2_q)

        hop3_q = self._safe_query(
            self.create_query_hop3,
            fallback="",
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            retrieved_titles=all_titles,
            prior_queries=prior_q,
        )
        if self._is_refusal(hop3_q):
            hop3_docs = []
        else:
            hop3_q = hop3_q.strip()
            hop3_docs = self.retrieve_k(hop3_q).passages
        hop_docs["hop3"] = hop3_docs
        all_titles = list(dict.fromkeys(all_titles + self._titles(hop3_docs)))

        # ---- HOP 4: independent gap analysis ----
        prior_q4 = list(prior_q)
        if not self._is_refusal(hop3_q):
            prior_q4.append(hop3_q)

        hop4_q = self._safe_query(
            self.create_query_hop4,
            fallback="",
            claim=claim,
            summary=summary_2,
            retrieved_titles=all_titles,
            prior_queries=prior_q4,
        )
        if self._is_refusal(hop4_q):
            hop4_docs = []
        else:
            hop4_q = hop4_q.strip()
            hop4_docs = self.retrieve_k(hop4_q).passages
        hop_docs["hop4"] = hop4_docs
        all_titles = list(dict.fromkeys(all_titles + self._titles(hop4_docs)))

        # ---- HARVEST: entity queries from ALL retrieved passages ----
        all_passages = []
        for docs in [hop1_docs, hop1b_docs, hop2_docs, hop3_docs, hop4_docs]:
            all_passages.extend(docs)

        harvest_queries = self._safe_harvest(claim, all_passages, all_titles)
        for i, hq in enumerate(harvest_queries[: self.harvest_query_cap]):
            hq = hq.strip()
            if hq and not self._is_refusal(hq):
                hq_docs = self.retrieve_k(hq).passages
                hop_docs[f"harvest_{i}"] = hq_docs
                all_titles = list(
                    dict.fromkeys(all_titles + self._titles(hq_docs))
                )

        # ---- BRIDGE: role/descriptor-implied entity ----
        all_passages_updated = []
        for docs in hop_docs.values():
            all_passages_updated.extend(docs)

        bridge_q = self._safe_bridge(claim, all_passages_updated, all_titles)
        if bridge_q:
            bridge_docs = self.retrieve_k(bridge_q).passages
            hop_docs["bridge"] = bridge_docs
            all_titles = list(
                dict.fromkeys(all_titles + self._titles(bridge_docs))
            )

        # ---- BUILD CANDIDATE POOL ----
        title_to_passage = {}
        for docs in hop_docs.values():
            for p in docs:
                t = p.split(" | ", 1)[0]
                if t not in title_to_passage:
                    title_to_passage[t] = p

        # Filter disambiguation pages
        candidate_titles = [
            t for t in title_to_passage if not self._is_disambig(t)
        ]

        if len(candidate_titles) <= self.final_doc_limit:
            final_docs = [title_to_passage[t] for t in candidate_titles]
            return dspy.Prediction(retrieved_docs=final_docs)

        # ---- RRF ----
        rrf_scores = self._compute_rrf(hop_docs)
        rrf_scores = {
            t: rrf_scores.get(t, 0.0)
            for t in candidate_titles
            if not self._is_disambig(t)
        }

        # RRF prefilter: keep top rrf_pool_cap by RRF
        rrf_sorted = sorted(candidate_titles, key=lambda t: rrf_scores.get(t, 0.0), reverse=True)
        pool_titles = rrf_sorted[: self.rrf_pool_cap]

        # ---- SAFETY-NET: top-1 per hop (hop1, hop2, hop3, hop4 only) ----
        safety_net = set()
        for hop_name in ["hop1", "hop2", "hop3", "hop4"]:
            docs = hop_docs.get(hop_name, [])
            if docs:
                top_t = docs[0].split(" | ", 1)[0]
                if not self._is_disambig(top_t):
                    safety_net.add(top_t)

        # ---- STAGE-1 reranker: score all pool candidates ----
        s1 = self._safe_rerank(self.rerank_stage1, claim, pool_titles)
        if s1 is None:
            s1 = [rrf_scores.get(t, 0.0) for t in pool_titles]

        need = self.final_doc_limit - len(safety_net)
        keep = min(max(need + self.stage1_keep_margin, 25), len(pool_titles))

        order_s1 = sorted(
            range(len(pool_titles)), key=lambda i: s1[i], reverse=True
        )[:keep]
        survivor_titles = [pool_titles[i] for i in order_s1]
        survivor_s1 = [s1[i] for i in order_s1]
        survivor_rrf = [
            rrf_scores.get(t, 0.0) for t in survivor_titles
        ]

        # ---- STAGE-2 reranker: re-score survivors ----
        s2 = self._safe_rerank(self.rerank_stage2, claim, survivor_titles)
        if s2 is None:
            s2 = list(survivor_s1)

        # ---- ENSEMBLE: 0.3*s1 + 0.4*s2 + 0.3*rrf ----
        ensemble = []
        for i in range(len(survivor_titles)):
            score = 0.3 * survivor_s1[i] + 0.4 * s2[i] + 0.3 * survivor_rrf[i]
            ensemble.append((survivor_titles[i], score))

        ensemble.sort(key=lambda x: x[1], reverse=True)

        # ---- BUILD FINAL 21: safety-net first, then ensemble fill ----
        final_titles = []
        final_seen = set()

        for t in safety_net:
            if t in title_to_passage and t not in final_seen:
                final_seen.add(t)
                final_titles.append(t)

        for t, _ in ensemble:
            if len(final_titles) >= self.final_doc_limit:
                break
            if t not in final_seen:
                final_seen.add(t)
                final_titles.append(t)

        # Pad from remaining pool if under 21
        if len(final_titles) < self.final_doc_limit:
            for t in pool_titles:
                if len(final_titles) >= self.final_doc_limit:
                    break
                if t not in final_seen:
                    final_seen.add(t)
                    final_titles.append(t)

        final_docs = [title_to_passage[t] for t in final_titles if t in title_to_passage]
        return dspy.Prediction(retrieved_docs=final_docs)