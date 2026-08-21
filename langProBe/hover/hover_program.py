import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_title(passage: str) -> str:
    """Normalize a 'title | text' passage down to its title key for dedup."""
    return re.sub(r"\s+", " ", passage.split(" | ")[0].strip().lower())


def _split_queries(text: str) -> list:
    """Parse a block of search queries (one per line, optionally numbered) from LM output."""
    if not text:
        return []
    parts = re.split(r"[\n\r]+", text)
    if len(parts) <= 1:
        parts = re.split(r"[;]+", text)
    out = []
    for ln in parts:
        ln = ln.strip()
        if not ln:
            continue
        # strip leading numbering / bullets / "Q1:" markers
        ln = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-\*\u2022]\s*|[Qq]\d*\s*[:\-]\s*)", "", ln)
        ln = ln.strip().strip("\"'“”‘’").strip()
        if ln:
            out.append(ln)
    return out


def _dedupe_queries(queries):
    seen = set()
    out = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _snippet(passage: str, n: int = 320) -> str:
    """Reduce a 'title | text' passage to a short snippet for the reranker context."""
    parts = passage.split(" | ", 1)
    if len(parts) == 2:
        title, text = parts
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > n:
            text = text[:n] + "..."
        return f"{title} | {text}"
    return re.sub(r"\s+", " ", passage).strip()[:n]


def _split_titles(text: str) -> list:
    """Parse a block of article titles (one per line, optionally numbered) from LM output."""
    if not text:
        return []
    out = []
    for ln in re.split(r"[\n\r]+", text):
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-\*\u2022]\s*)", "", ln).strip()
        ln = ln.strip("\"'“”‘’").strip()
        if ln:
            out.append(ln)
    return out


class Decompose(dspy.Signature):
    """You are a Wikipedia retrieval expert. A factual claim requires several Wikipedia
    articles to verify. Write a set of DIVERSE search queries to retrieve those articles.

    Rules:
    - Each query must target a distinct entity, proper noun, organization, work, place, or
      event mentioned in the claim.
    - Also write queries for entities IMPLIED but not named in the claim that would be
      needed to verify it (the multi-hop connections).
    - Use concise keyword queries (entity name + a short descriptor), NOT full sentences.
    - Make every query DISTINCT; do not repeat the same entity combination.
    - Write 4 to 6 queries, exactly one per line."""

    claim = dspy.InputField()
    search_queries = dspy.OutputField(
        desc="4 to 6 distinct Wikipedia search queries, one per line"
    )


class ReasonGap(dspy.Signature):
    """You are verifying a factual claim by retrieving Wikipedia articles. You have already
    retrieved some Wikipedia passages, and the titles of ALL retrieved articles are listed in
    retrieved_titles. Your job is to find the supporting articles that are STILL MISSING and
    query them.

    Think step-by-step:
    1. List the DISTINCT entities, people, works, places, events, and facts that the claim
       REQUIRES to be verified (the full multi-hop chain, including entities IMPLIED but not
       named in the claim).
    2. For each required entity, check the retrieved_titles list (and the passage text) to
       decide whether a DEDICATED Wikipedia article about it has already been retrieved.
       IMPORTANT: an entity merely MENTIONED in the text of another article is NOT the same
       as having its own retrieved article. If the entity's own article is not in
       retrieved_titles, it is MISSING and must be queried, even if its name appears in some
       other passage's text.
    3. Identify the MISSING pieces: required entities NAMED LITERALLY in the passage text
       whose own article is not in retrieved_titles (e.g. a co-star, an author, a hometown,
       a work title shown in quotes, an event like "Death of David Bowie"), OR entities
       implied by the claim not yet retrieved. Treat an event, a death, a disambiguated
       work/season, or any distinct Wikipedia-notable concept as its OWN article.
    4. Output one concise Wikipedia search query per MISSING entity, up to 3, targeting the
       most important missing pieces. Use each entity's OWN article title; use the Wikipedia
       disambiguated-title style in parentheses when ambiguous (e.g. "The Grapes of Wrath
       (film)", "The Secret Agent (TV series)", "Deutschland sucht den Superstar (season 9)",
       "Shim Ji-ho", "Ron Teachworth"). Do NOT combine two entities into one relational query
       (e.g. avoid "John Arledge John Ford film"); query each entity's own article separately.

    Do NOT target entities already in prior_queries or whose article is already in
    retrieved_titles. Write 1 to 3 NEW queries, exactly one per line. If and only if every
    required entity already has its own article in retrieved_titles, output NONE."""

    claim = dspy.InputField()
    passages = dspy.InputField(
        desc="Full text of the top retrieved Wikipedia passages so far, each 'title | text'"
    )
    retrieved_titles = dspy.InputField(
        desc="All Wikipedia article titles retrieved so far, one per line"
    )
    prior_queries = dspy.InputField(
        desc="Search queries already issued, one per line"
    )
    thought = dspy.OutputField(
        desc="Step-by-step gap analysis: what the claim requires, what is found (in retrieved_titles), what is missing"
    )
    search_queries = dspy.OutputField(
        desc="1 to 3 NEW Wikipedia search queries for missing entities, one per line, or NONE"
    )


class Rerank(dspy.Signature):
    """You are selecting the Wikipedia articles that best support verifying a factual claim.

    You are given a factual claim and a list of candidate Wikipedia articles (title | short
    snippet). Choose which of these candidates are most likely to be REQUIRED supporting
    articles for the claim, and order them.

    Recall is all-or-nothing: the claim requires ALL of its supporting articles to be
    present in the final set. An article is a STRONG match if it is the DEDICATED Wikipedia
    article for a distinct entity, person, work, place, or event that the claim depends on
    (named in the claim OR reachable via a multi-hop connection, e.g. a co-creator, a
    hometown, a parent work, an event). An article that merely MENTIONS a required entity in
    passing is a WEAK match; prefer the entity's OWN dedicated article.

    Output the candidate article TITLES, one per line, MOST relevant first. Include every
    article likely to be a required supporting article; omit articles clearly unrelated to
    the claim. Copy the title exactly as given. """

    claim = dspy.InputField()
    candidates = dspy.InputField(
        desc="Candidate Wikipedia articles to rank, one per line, 'title | snippet'"
    )
    ranked = dspy.OutputField(
        desc="Candidate article titles, one per line, most relevant first"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval for a claim, optimized for document recall.

    Pipeline:
      1. Decompose the claim into several diverse search queries (one per entity/aspect).
      2. Retrieve a large candidate set per query (high k) and fuse across queries with
         Reciprocal Rank Fusion (RRF), deduplicating by document title.
      3. Gap-analysis-driven multi-hop expansion: reason about what the claim requires,
         which required entities already have a retrieved article, and which are STILL
         MISSING; then issue up to 3 targeted follow-up queries for the missing entities.
          Repeat for a second round so entities surfaced by round 1's new docs can themselves
          be queried, enabling deep multi-hop chains (e.g. film -> director -> hometown).
       4. LM list-reranker: the baseline top-21 (guaranteed hits + top RRF fill) is protected;
          the reranker may only PROMOTE buried candidates (ranked below top-21) by displacing
          the weakest-RRF fill docs (rescues golds retrieved but buried below top-21, with zero
          extra searches and no demotion of high-RRF golds). Return the top 21 unique documents.

    EVALUATION
    - This system is assessed by retrieving ALL the correct supporting documents.
    - The system must provide at most 21 documents at the end of the program.'''

    MAX_DOCS = 21
    RRF_K = 60          # standard RRF constant
    K_CLAIM = 25        # broadest query, gets the largest retrieval budget
    K_QUERY = 20        # per-query retrieval budget for generated queries
    # Per-query top-N guarantee: the top-N docs from EACH query are guaranteed a slot
    # in the final output. This protects high-rank docs that are surfaced by only one
    # query (e.g. an exact-title multi-hop entity) from being buried under documents
    # that appear in many queries but at lower ranks. RRF alone (k=60) over-rewards
    # cross-query coverage and can drop these single-query hits below the top-21.
    GUARANTEE_N = 2
    # Gap-analysis expansion rounds. Each round reasons over the current top passages +
    # the retrieved_titles list and issues up to QUERIES_PER_ROUND targeted follow-up
    # queries for still-missing supporting entities (penalty is negligible vs. recall).
    EXPAND_ROUNDS = 2
    QUERIES_PER_ROUND = 3
    # LM list-reranker (final stage). The architecture's baseline output (per-query
    # GUARANTEE_N=2 hits + the top non-guaranteed docs by RRF) is PROTECTED: those 21 docs are
    # never removed. The reranker only gets to PROMOTE BURIED candidates (docs ranked below the
    # top-21 by RRF) into the final set, by displacing the WEAKEST-RRF non-guaranteed fill docs.
    # This is the "only promote, never demote high-RRF" safety from iter-4 memory: it rescues
    # golds that were retrieved but buried below the top-21 (the burial failure mode) while
    # protecting the golds that already ride the RRF fill (which a full rerank would demote).
    # It adds ZERO ColBERT searches (no new docs enter the RRF pool) -> no burial risk.
    RERANK_POOL_SIZE = 50
    RERANK_SWAPS = 7       # max buried docs the reranker may promote (weakest fill displaced)

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=self.K_QUERY)
        self.decompose = dspy.ChainOfThought(Decompose)
        self.reason_gap = dspy.ChainOfThought(ReasonGap)
        self.rerank = dspy.ChainOfThought(Rerank)

    def _retrieve_and_fuse(self, query, k, ranks, best_rank, passages, guaranteed):
        """Retrieve docs for one query and accumulate RRF scores + representative passages.

        Also marks the top-GUARANTEE_N titles as guaranteed a final slot."""
        try:
            retrieved = self.retrieve(query, k=k).passages
        except Exception:
            retrieved = []
        for rank, psg in enumerate(retrieved):
            title = _normalize_title(psg)
            if not title:
                continue
            ranks[title] = ranks.get(title, 0.0) + 1.0 / (self.RRF_K + rank)
            # keep the passage from the highest-ranked (smallest rank) occurrence
            if title not in best_rank or rank < best_rank[title]:
                best_rank[title] = rank
                passages[title] = psg
            if rank < self.GUARANTEE_N:
                guaranteed.add(title)

    def forward(self, claim):
        ranks = {}            # title -> accumulated RRF score
        best_rank = {}        # title -> best (smallest) rank seen
        passages = {}         # title -> representative 'title | text' passage
        guaranteed = set()    # titles guaranteed a slot (top-N of some query)

        # ---- Stage 1: claim + decomposed queries ----
        queries = [claim]
        try:
            dec = self.decompose(claim=claim).search_queries
            queries += _split_queries(dec)
        except Exception:
            pass
        queries = _dedupe_queries(queries)[:6]

        # claim gets the largest budget; generated queries get the standard budget
        self._retrieve_and_fuse(queries[0], self.K_CLAIM, ranks, best_rank, passages, guaranteed)
        for q in queries[1:]:
            self._retrieve_and_fuse(q, self.K_QUERY, ranks, best_rank, passages, guaranteed)

        # ---- Stage 2: gap-analysis-driven multi-hop expansion ----
        # Each round feeds the LM the FULL TEXT of the current top fused passages PLUS the
        # complete retrieved_titles list (so it can tell "mentioned in text" apart from "has
        # its own article") and asks for up to 3 targeted follow-up queries for STILL-MISSING
        # supporting entities. Round 2 sees docs newly retrieved in round 1, enabling deeper
        # multi-hop chains. Stops a round early if the LM reports nothing missing (NONE).
        all_queries = list(queries)
        prior = set(q.lower() for q in queries)
        for _ in range(self.EXPAND_ROUNDS):
            try:
                top_titles = sorted(ranks, key=lambda t: -ranks[t])[:15]
                top_passages = [passages[t] for t in top_titles if t in passages]
                if not top_passages:
                    break
                all_titles = sorted(ranks, key=lambda t: -ranks[t])
                rg = self.reason_gap(
                    claim=claim,
                    passages="\n\n".join(top_passages),
                    retrieved_titles="\n".join(all_titles[:30]),
                    prior_queries="\n".join(all_queries),
                )
                follow_ups = _dedupe_queries(_split_queries(rg.search_queries))
                follow_ups = [q for q in follow_ups if q.upper() != "NONE"]
                follow_ups = [q for q in follow_ups if q.lower() not in prior][: self.QUERIES_PER_ROUND]
                if not follow_ups:
                    break
                for q in follow_ups:
                    self._retrieve_and_fuse(q, self.K_QUERY, ranks, best_rank, passages, guaranteed)
                all_queries.extend(follow_ups)
                prior.update(q.lower() for q in follow_ups)
            except Exception:
                break

        # ---- Final: protected baseline top-21, with LM-reranker promoting buried candidates ----
        # Baseline output (PROTECTED, never removed): guaranteed docs (by best rank) + the top
        # non-guaranteed docs by RRF. The reranker may only PROMOTE buried docs (ranked below
        # this baseline top-21) by displacing the WEAKEST-RRF non-guaranteed fill docs.
        guaranteed_ordered = sorted(
            guaranteed, key=lambda t: best_rank.get(t, 1_000_000)
        )
        pool_order = sorted(ranks, key=lambda t: -ranks[t])  # all titles, RRF desc
        fill_budget = self.MAX_DOCS - len(guaranteed_ordered)
        # fill_by_rrf is in RRF-desc order; the TAIL is the weakest-RRF fill (displaceable).
        fill_by_rrf = [t for t in pool_order if t not in guaranteed][:fill_budget]
        fill_seen = set(fill_by_rrf)
        buried = [
            t for t in pool_order
            if t not in guaranteed and t not in fill_seen
        ][: self.RERANK_POOL_SIZE]
        if fill_budget > 0 and buried:
            try:
                candidates = [_snippet(passages[t]) for t in buried if t in passages]
                rr = self.rerank(claim=claim, candidates="\n".join(candidates))
                buried_set = set(buried)
                promoted = []
                promoted_set = set()
                for title_str in _split_titles(rr.ranked):
                    t = _normalize_title(title_str)
                    if not (t and t in buried_set and t not in promoted_set):
                        continue
                    # Displace the weakest-RRF fill doc not already displaced/promoted.
                    # Iterate the tail (weakest RRF) first; guaranteed docs are not in fill_by_rrf.
                    victim_idx = None
                    for i in range(len(fill_by_rrf) - 1, -1, -1):
                        if fill_by_rrf[i] not in promoted_set and fill_by_rrf[i] not in guaranteed:
                            victim_idx = i
                            break
                    if victim_idx is None:
                        break
                    fill_by_rrf[victim_idx] = t  # swap the promoted buried doc in
                    promoted.append(t)
                    promoted_set.add(t)
                    if len(promoted) >= self.RERANK_SWAPS:
                        break
            except Exception:
                pass
        ordered = guaranteed_ordered + fill_by_rrf
        final = [passages[t] for t in ordered[: self.MAX_DOCS] if t in passages]
        return dspy.Prediction(retrieved_docs=final)
