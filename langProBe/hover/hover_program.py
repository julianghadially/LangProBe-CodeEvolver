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
       W", "starring X and Y", "cast includes Z") still need their OWN Wikipedia
       article retrieved to verify or refute the claim. This is multi-hop
       retrieval: a passage often names the next-hop entity to chase.
    3. When passages describe a film, TV, music video, album or other *work*,
       ALWAYS list the cast, crew, director, producer, writer and any persons
       named in the work's article (including the lead actors/actresses) — these
       named people are the highest-value next-hop targets in multi-hop claims.
       Do NOT fixate on the venue, festival, or broadcaster where the work
       premiered; that is a connector, not the entity the claim links to.
    4. Notes any entity that has already been covered vs. one still missing.

    Do not omit named entities that appear in the passages even if they seem
    tangential; the next hop will use them as search targets."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    summary: str = dspy.OutputField(desc="Concise entity-focused summary of passages, listing named cast/crew and still-missing entities")


class SummarizeHop2(dspy.Signature):
    """You are retrieving Wikipedia documents to verify a multi-hop claim and
    have already completed two retrieval hops.

    Given the claim, the prior summary, the new passages, and the list of
    Wikipedia article titles already retrieved across hops 1 and 2, write an
    updated concise factual summary that:
    1. Integrates the new passages with the prior summary, tracking which
       supporting entities have now been found and which are still missing.
       An entity counts as "found" ONLY if its exact Wikipedia article title
       appears in `retrieved_titles`.
    2. Emphasizes any named entity mentioned in the passages whose OWN
       Wikipedia article has not yet been retrieved (its title is NOT in
       `retrieved_titles`) and is needed to verify or refute the claim
       (multi-hop chain following).
    3. When passages describe a film, TV, music video, album or other *work*,
       list its cast, crew, director, producer, writer and named
       actors/actresses — these named people are the highest-value next-hop
       targets. Do NOT fixate on the venue/festival/broadcaster; it is a
       connector, not the entity the claim links to.
    4. Explicitly lists the entity that should be searched next and why, chosen
       from entities still missing (title not in retrieved_titles).

    Be precise about entity names; they will become search queries."""
    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="Summary from the prior hop")
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hops 1 and 2")
    summary: str = dspy.OutputField(desc="Updated entity-focused summary, listing missing cast/crew and the entity to search next")


class CreateQueryHop2(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Hop 1 retrieved passages using the raw claim and they were summarized. Now
    generate ONE search query for hop 2 that follows the multi-hop chain:
    - Pick a named entity that appears in the summary (often introduced by the
      hop-1 passages via a relational phrase such as "produced by X",
      "dedicated to Y", "located beside Z", "directed by W", "starring X") but
      whose own Wikipedia article title is NOT in `retrieved_titles`.
    - The query should target that entity's Wikipedia article. Prefer the bare
      entity name; append a single disambiguator (e.g. "film", "band",
      "actor", "company", "place") only when the name is genuinely ambiguous.
    - Prefer a PERSON (a named person) over a connector entity (festival, TV
      series, broadcaster, venue) when the claim links the subject to a person.
    - `prior_queries` lists queries already issued; do NOT duplicate them, do
      NOT restate the whole claim, do NOT refuse, and do NOT answer "none"/"no
      query". If your first choice was already issued, pick a different missing
      entity.

    Output a single concise search query (the bare entity name plus optional
    disambiguator only)."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved in hop 1")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class CreateQueryHop3(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Two hops of retrieval have been completed. Generate ONE search query for
    hop 3 that follows the multi-hop chain to an entity still missing:
    - From the summaries, identify a named entity mentioned by the retrieved
      passages (e.g. via "produced by X", "dedicated to Y", "directed by W",
      "located beside Z", "starring X and Y") whose own Wikipedia article
      title is NOT in `retrieved_titles` and is needed to verify or refute the
      claim.
    - Target that entity's Wikipedia article. Prefer the bare entity name;
      append a single disambiguator only when the name is genuinely ambiguous.
    - Prefer a PERSON over a connector entity (festival, venue, broadcaster,
      TV series) when the claim links the subject to a person.
    - Cast/crew-follow: when a film, TV, music video, album or other work
      article was retrieved and the claim concerns a person in that work
      (cast, director, producer, writer, named actor), query the named
      cast/crew member, NOT the venue/festival where it premiered.
    - Pick a DIFFERENT entity than hop 2 if multiple are missing. `prior_queries`
      lists queries already issued; do NOT duplicate them, do NOT restate the
      whole claim, do NOT refuse, and do NOT answer "none"/"no query".

    Output a single concise search query (the bare entity name plus optional
    disambiguator only)."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    summary_2: str = dspy.InputField(desc="Updated entity-focused summary after hop 2")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hops 1 and 2")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class HarvestEntityQueries(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim. Several
    retrieval hops have completed and their Wikipedia abstract passages are
    listed below.

    Your job: find NAMED ENTITIES (people, organizations, places, films/works,
    events, teams, games) that are MENTIONED BY NAME inside these passages but
    whose OWN Wikipedia article has NOT yet been retrieved (its title is NOT in
    `retrieved_titles`). These are the chain-follow targets the claim may link to
    — a passage often textually names the next-hop entity to chase (e.g.
    "songwriter Rosi Golan", "founded by Charlotte Baldwin Allen", "the city of
    Rochester Hills", "the video game F.E.A.R."). Issue search queries to pull
    the OWN articles of the most claim-relevant such entities.

    Rules:
    - Only entities whose name actually appears in the passage text. Do NOT
      invent entities that are merely implied by the claim but never named in a
      passage.
    - Exclude any entity whose article title is already in `retrieved_titles` or
      whose name matches a query already in `prior_queries`.
    - Prefer entities the claim's chain links to (the subject's spouse, founder,
      director, partner, home city, parent work, or named role) over background
      mentions. A person/place/work that a retrieved passage introduces via a
      relational phrase ("married X", "born in Y", "composed by Z", "directed
      by W", "founded in L") is highest value.
    - Prefer a PERSON over a connector (venue, broadcaster, festival) when the
      claim links the subject to a person.
    - Output BARE entity names, adding a single disambiguator (e.g. "film",
      "band", "actor", "place") only when the name is genuinely ambiguous.

    `prior_queries` lists queries already issued; do NOT duplicate them. Do not
    restate the whole claim, refuse, or answer "none". Output up to 3 distinct
    search queries."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="All Wikipedia abstract passages retrieved across earlier hops")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued")
    queries: list[str] = dspy.OutputField(desc="Up to 3 bare-entity-name search queries for mentioned-but-unretrieved entities, most claim-relevant first")


class BridgeDescriptorQuery(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim, as a
    final BRIDGE hop. Several retrieval hops and an entity-harvest sweep have
    completed, yet some entities the claim links to may STILL be missing because
    NO retrieved passage names them — they are only IMPLIED by a role, title,
    location, or descriptor in the claim or in the retrieved passages.

    Your job: NAME a single real-world entity (person, organization, place, or
    work) that the claim's chain implies by ROLE / title / location / descriptor
    but that is NOT named in any retrieved passage and whose own Wikipedia
    article is NOT in `retrieved_titles`. Then output a search query for it.

    Examples of role/descriptor-implied bridges:
    - "the founding director of the X Centre" -> the person who founded/directs it
    - "starred in a film directed by Shane Meadows" -> the actor of that film
    - "the subsidiary of Comair" -> the airline Comair owns
    - "the Nigerian midfielder who played in match Y" -> the specific player
    - "a film directed by Z" -> the lead actor or the film itself

    Rules:
    - Only emit an entity genuinely IMPLIED by a relational phrase in the claim
      or passages; do NOT invent entities the claim does not point to.
    - The entity must NOT be in `retrieved_titles` and a query for it must NOT
      be in `prior_queries`.
    - Prefer the bare entity name; add a single disambiguator (e.g. "actor",
      "footballer", "tv series", "place") only when ambiguous.
    - Do NOT restate the whole claim, do NOT refuse, and do NOT answer
      "none"/"no query". If multiple entities are implied, pick the one most
      central to verifying the claim.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_2: str = dspy.InputField(desc="Entity-focused summary of the chain so far")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued")
    query: str = dspy.OutputField(desc="A single search query for one role/descriptor-implied missing entity")


class CreateQueryHop4(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim, as a
    final independent GAP-ANALYSIS hop after three hops have been completed.

    Answer ONE question: which single entity, needed to verify or refute the
    claim, has NOT yet had its own Wikipedia article retrieved and should be
    searched now? Then output a search query for that entity.
    - Use `summary_2`, the list of `retrieved_titles` (titles already retrieved
      across hops 1-3), and `prior_queries` (queries already issued).
    - An entity is "still missing" only if its title is NOT in
      `retrieved_titles` and a query for it is NOT in `prior_queries`.
    - Cast/crew-follow: when a film, TV, music video, album or other work
      article was retrieved and the claim concerns a person in that work
      (cast, director, producer, writer, named actor), query the named
      cast/crew member's own article, NOT the venue/festival/broadcaster where
      the work premiered (that is a connector, not the linked entity).
    - Prefer a PERSON over a connector entity (festival, venue, broadcaster,
      TV series) when the claim links the subject to a person.
    - Do NOT duplicate prior_queries, do NOT restate the whole claim, do NOT
      refuse, and do NOT answer "none"/"no query". If every obvious entity has
      already been retrieved, pick the next-most-likely missing person named
      in the summaries.

    Output a single concise search query (the bare entity name plus optional
    disambiguator only)."""
    claim: str = dspy.InputField()
    summary_2: str = dspy.InputField(desc="Updated entity-focused summary after hop 2")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hops 1-3")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one still-missing entity")


class RerankDocs(dspy.Signature):
    """You are verifying a multi-hop Wikipedia claim. A pool of candidate
    Wikipedia abstract passages has been retrieved across several search hops.
    Score EACH passage 0-10 for how relevant its Wikipedia article is to
    verifying the claim's multi-hop chain.

    Score 9-10: this passage's Wikipedia article is one of the entities the
    claim directly links to (the subject entity, or a person/organization/place
    /work that a relational phrase in the claim or in retrieved passages points
    to — e.g. "directed by X", "produced by Y", "starring Z", "founded by W",
    "located in L"). Gold supporting articles are in this band.
    Score 5-8: clearly on-topic and plausibly part of the chain.
    Score 1-4: only loosely related (same broad field, a neighbor article, a
    disambiguator); a surface-token hit, not the linked entity.
    Score 0: off-topic distractor.

    A film/album/show article itself is highly relevant when the claim concerns
    that work; a cast/crew member's own biography article is highly relevant
    when the claim links the subject to that person. A venue/festival/broadcaster
    is a low-value connector unless the claim literally links to it.

    Output one integer score per passage, in the SAME order as the input
    passages. Output exactly len(passages) scores."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Candidate Wikipedia abstract passages to score")
    scores: list[int] = dspy.OutputField(desc="One relevance score 0-10 per passage, in input order")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15
        self.k_hop4 = 15
        self.harvest_k = 15
        self.final_doc_limit = 21
        self.harvest_q_cap = 2
        self.create_query_hop2 = dspy.ChainOfThought(CreateQueryHop2)
        self.create_query_hop3 = dspy.ChainOfThought(CreateQueryHop3)
        self.create_query_hop4 = dspy.ChainOfThought(CreateQueryHop4)
        self.harvest = dspy.ChainOfThought(HarvestEntityQueries)
        self.bridge = dspy.ChainOfThought(BridgeDescriptorQuery)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.retrieve_k_hop4 = dspy.Retrieve(k=self.k_hop4)
        self.summarize1 = dspy.ChainOfThought(SummarizeHop1)
        self.summarize2 = dspy.ChainOfThought(SummarizeHop2)
        self.rerank = dspy.Predict(RerankDocs)

    @staticmethod
    def _titles(passages):
        return [p.split(" | ", 1)[0] for p in passages]

    @staticmethod
    def _dedup(passages):
        """Deduplicate passages by their leading title ('Title | ...'),
        preserving first-seen order."""
        seen = set()
        unique = []
        for p in passages:
            title = p.split(" | ", 1)[0]
            if title not in seen:
                seen.add(title)
                unique.append(p)
        return unique

    @staticmethod
    def _rerank_pool(passages_groups, claim, rerank_fn, limit, safety_count=0):
        """Build a title-deduped candidate pool from per-hop passage lists and
        return the top-`limit` passages via a TWO-STAGE pointwise rerank:

          Stage 1 (coarse): score every candidate, keep the top
            `stage1_keep` (~limit + 9) by score -- this drops the bulk of
            distractors so survivor golds become salient.
          Stage 2 (fine):   re-score only the stage-1 survivors and keep the
            top `need` by the stage-2 score.

        Two stages absorb enlarged harvest/bridge pools that the single-pass
        scorer diluted (a true gold that survives stage 1 gets a cleaner, less
        noisy re-score among a high-relevance-only pool). Ties break by
        earliest hop then earliest rank, preserving FIFO priority for
        primary-hop golds. On any LM parse/length failure we fall back to
        hop/rank FIFO order.

        `safety_count`: each of the first `safety_count` hops has its
        first-seen (rank-0) passage automatically guaranteed a final slot,
        ahead of the score-ranked fill. This protects primary-hop golds that
        the pointwise reranker occasionally false-drops below distractors."""
        seen_titles = set()
        guaranteed = []
        candidates = []  # (passage, hop, rank)
        for hop, group in enumerate(passages_groups):
            ranked_unique = []
            for p in group:
                title = p.split(" | ", 1)[0]
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                ranked_unique.append(p)
            for rank, p in enumerate(ranked_unique):
                candidates.append((p, hop, rank))
            if hop < safety_count and ranked_unique:
                guaranteed.append(ranked_unique[0])
        if not candidates:
            return []
        if len(candidates) <= limit:
            return [p for p, _, _ in candidates]
        guaranteed_titles = {p.split(" | ", 1)[0] for p in guaranteed}
        scored = [c for c in candidates if c[0].split(" | ", 1)[0] not in guaranteed_titles]
        kept = list(guaranteed)
        need = limit - len(kept)
        if need <= 0:
            return kept[:limit]
        if not scored:
            return kept[:limit]
        # ---- Stage 1 (coarse): score every candidate, keep the top stage1_keep
        stage1_keep = min(len(scored), max(need + 9, 25))
        s1 = HoverMultiHop._score_candidates(rerank_fn, claim, scored)
        if s1 is None:
            return (kept + [c[0] for c in scored])[:limit]
        order1 = sorted(
            range(len(scored)),
            key=lambda i: (-s1[i], scored[i][1], scored[i][2]),
        )
        surv_idxs = order1[:stage1_keep]
        survivors = [scored[i] for i in surv_idxs]
        if len(survivors) <= need:
            kept.extend(c[0] for c in survivors)
            return kept[:limit]
        # ---- Stage 2 (fine): re-score only the survivors, keep top `need`
        s2 = HoverMultiHop._score_candidates(rerank_fn, claim, survivors)
        if s2 is None:
            kept.extend(survivors[i][0] for i in range(need))
            return kept[:limit]
        order2 = sorted(
            range(len(survivors)),
            key=lambda i: (-s2[i], survivors[i][1], survivors[i][2]),
        )
        kept.extend(survivors[i][0] for i in order2[:need])
        return kept[:limit]

    @staticmethod
    def _score_candidates(rerank_fn, claim, scored):
        """Run the pointwise reranker over `scored` [(passage,hop,rank),...].
        Returns int scores aligned to `scored`, or None on any
        parse/length-mismatch failure."""
        passages = [c[0] for c in scored]
        try:
            out = rerank_fn(claim=claim, passages=passages)
            scores = list(out.scores)
        except Exception:
            return None
        if not scores or len(scores) != len(scored):
            return None
        try:
            return [int(s) for s in scores]
        except Exception:
            return None

    @staticmethod
    def _is_refusal(query):
        if query is None:
            return True
        q = query.strip().lower()
        if q == "" or q in ("none", "no query", "no", "n/a", "no query needed"):
            return True
        if q.startswith(("no query", "none of", "no ", "i don't", "i do not")):
            return True
        return False

    def _gen_query(self, predictor, **kwargs):
        """Run a ChainOfThought query generator, returning a single query
        string. Falls back to an empty string (skip the hop) on any parse or
        missing-field failure so a single LM crash cannot zero an example."""
        try:
            out = predictor(**kwargs)
            q = getattr(out, "query", None)
            if q is None or not str(q).strip():
                return ""
            return str(q).strip()
        except Exception:
            return ""

    def _fifo_then_interleave_dedup(self, primary_docs, late_docs_groups, limit):
        """Primary hops (1, 2) keep FIFO priority for full tail coverage; late
        hops (3, 4) round-robin into remaining tail slots so their gap-analysis
        golds survive the 21-doc cap."""
        primary = self._dedup(primary_docs)
        if len(primary) >= limit:
            return primary[:limit]
        seen = set(self._titles(primary))
        late_unique_groups = []
        for group in late_docs_groups:
            ug = []
            for p in group:
                t = p.split(" | ", 1)[0]
                if t not in seen and t not in {x.split(" | ", 1)[0] for x in ug}:
                    ug.append(p)
            late_unique_groups.append(ug)
        capacity = limit - len(primary)
        result = list(primary)
        # round-robin across late groups until cap reached
        filled = 0
        while filled < capacity:
            added_any = False
            for group in late_unique_groups:
                if group:
                    p = group.pop(0)
                    result.append(p)
                    seen.add(p.split(" | ", 1)[0])
                    filled += 1
                    added_any = True
                    if filled >= capacity:
                        break
            if not added_any:
                break
        return result[:limit]

    def forward(self, claim):
        prior_queries = [claim]

        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary
        retrieved_titles = self._titles(hop1_docs)

        # HOP 2
        hop2_query = self._gen_query(
            self.create_query_hop2,
            claim=claim,
            summary_1=summary_1,
            retrieved_titles=retrieved_titles,
            prior_queries=prior_queries,
        )
        if self._is_refusal(hop2_query):
            hop2_docs = []
        else:
            prior_queries.append(hop2_query)
            hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim,
            context=summary_1,
            passages=hop2_docs,
            retrieved_titles=retrieved_titles + self._titles(hop2_docs),
        ).summary
        retrieved_titles = self._titles(hop1_docs) + self._titles(hop2_docs)

        # HOP 3
        hop3_query = self._gen_query(
            self.create_query_hop3,
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            retrieved_titles=retrieved_titles,
            prior_queries=prior_queries,
        )
        if self._is_refusal(hop3_query):
            hop3_docs = []
        else:
            prior_queries.append(hop3_query)
            hop3_docs = self.retrieve_k(hop3_query).passages
        retrieved_titles = retrieved_titles + self._titles(hop3_docs)

        # HOP 4 (independent gap-analysis; does NOT re-summarize into chain)
        hop4_query = self._gen_query(
            self.create_query_hop4,
            claim=claim,
            summary_2=summary_2,
            retrieved_titles=retrieved_titles,
            prior_queries=prior_queries,
        )
        if self._is_refusal(hop4_query):
            hop4_docs = []
        else:
            hop4_docs = self.retrieve_k_hop4(hop4_query).passages

        # HARVEST: chain-follow sweep over entity names mentioned in retrieved
        # passage text whose own article is not yet retrieved. Targets the
        # recurring mode where a gold (songwriter, spouse, home city, parent
        # game) is named inside an already-retrieved passage but never had its
        # own article queried. Independent of hops 1-4; does not write back.
        harvest_docs = []
        all_passages = hop1_docs + hop2_docs + hop3_docs + hop4_docs
        if all_passages:
            try:
                harvested = self.harvest(
                    claim=claim,
                    passages=all_passages,
                    retrieved_titles=retrieved_titles + self._titles(hop4_docs),
                    prior_queries=prior_queries,
                ).queries
            except Exception:
                harvested = []
            if not isinstance(harvested, list):
                harvested = [harvested] if harvested else []
            issued = set(q.strip().lower() for q in prior_queries)
            count = 0
            for q in harvested:
                if count >= self.harvest_q_cap:
                    break
                if not isinstance(q, str):
                    continue
                qs = q.strip()
                if not qs or self._is_refusal(qs):
                    continue
                if qs.lower() in issued:
                    continue
                issued.add(qs.lower())
                prior_queries.append(qs)
                harvest_docs.extend(self.retrieve_k(qs).passages)
                count += 1

        bridge_docs = []
        bridge_titles_before = retrieved_titles + self._titles(hop4_docs) + self._titles(harvest_docs)
        bridge_query = self._gen_query(
            self.bridge,
            claim=claim,
            summary_2=summary_2,
            retrieved_titles=bridge_titles_before,
            prior_queries=prior_queries,
        )
        if not self._is_refusal(bridge_query):
            prior_queries.append(bridge_query)
            bridge_docs = self.retrieve_k_hop4(bridge_query).passages

        all_docs = self._rerank_pool(
            [hop1_docs, hop2_docs, hop3_docs, hop4_docs, harvest_docs, bridge_docs],
            claim,
            self.rerank,
            self.final_doc_limit,
            safety_count=4,
        )
        return dspy.Prediction(retrieved_docs=all_docs)