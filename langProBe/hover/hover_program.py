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
    retrieved some Wikipedia passages. Your job is to find what is STILL MISSING and retrieve
    one missing piece at a time.

    Think step-by-step:
    1. List the DISTINCT entities, people, works, places, events, and facts that the claim
       requires in order to be verified (the full multi-hop chain, including entities that
       are IMPLIED but not named in the claim).
    2. For each required entity, decide whether the retrieved passages already contain a
       DEDICATED Wikipedia article about it - check both the passage TITLES and the passage
       TEXT for an article whose title is exactly that entity.
    3. Identify the MISSING pieces: required entities that are NAMED LITERALLY in the passage
       text but do NOT yet have their own retrieved article (e.g. a co-star, an author, a
       hometown, a work title in quotes, an event like "Death of David Bowie"), OR entities
       implied by the claim that have not been retrieved at all. Treat an event, a death, a
       disambiguated work/season, or any distinct Wikipedia-notable concept as its OWN article.
    4. Pick the SINGLE most important missing entity and output ONE concise Wikipedia search
       query for it. Use the entity's OWN article title. Use the Wikipedia disambiguated-title
       style in parentheses when the name is ambiguous (e.g. "The Grapes of Wrath (film)",
       "The Secret Agent (TV series)", "Deutschland sucht den Superstar (season 9)",
       "Shim Ji-ho", "Ron Teachworth"). Do NOT combine two entities into one relational
       query (e.g. avoid "John Arledge John Ford film"); query each entity's own article.

    If and only if EVERY distinct entity required to verify the claim already has its own
    retrieved Wikipedia article among the retrieved titles, output action STOP. Otherwise
    output exactly one new search query (NOT already in prior_queries). When in doubt,
    propose another query rather than stopping, because a missing supporting article is
    worse than an extra search.

    Prior queries already issued are listed; do NOT repeat them."""

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
        desc="Step-by-step gap analysis: what the claim requires, what is found, what is missing"
    )
    action = dspy.OutputField(
        desc="Either 'STOP' or a single concise Wikipedia search query for the most important missing entity"
    )


def _parse_action(text):
    """Parse a ReasonGap action into 'STOP' or a single query string (or None)."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip().strip("\"'“”‘’")
    if not line:
        return None
    if line.upper().startswith("STOP"):
        return "STOP"
    return line


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval for a claim, optimized for document recall.

    Pipeline:
      1. Decompose the claim into several diverse search queries (one per entity/aspect).
      2. Retrieve a large candidate set per query (high k) and fuse across queries with
         Reciprocal Rank Fusion (RRF), deduplicating by document title.
      3. ReAct-style adaptive gap analysis: repeatedly reason about what the claim
         requires, which required entities are already retrieved, and which are STILL
         MISSING, then issue ONE targeted follow-up query for the most important missing
         entity; retrieve and fuse. Newly retrieved docs become visible to the next
         reasoning step, enabling arbitrarily deep multi-hop chains (e.g. film -> director
         -> hometown).
      4. Return the top 21 unique documents by fused score.

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
    # ReAct loop budget: number of adaptive gap-analysis follow-up queries. Each step
    # reasons over the current top passages and issues one targeted query, so this is
    # also the max extra searches (penalty is negligible vs. recovering a missing gold).
    REACT_STEPS = 7

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=self.K_QUERY)
        self.decompose = dspy.ChainOfThought(Decompose)
        self.reason_gap = dspy.ChainOfThought(ReasonGap)

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

        # ---- Stage 2: ReAct-style adaptive gap-analysis retrieval ----
        # Each step reasons over the FULL TEXT of the current top fused passages (so specific
        # entity names are preserved) plus the full list of retrieved titles and prior queries,
        # then issues ONE targeted query for the most important MISSING supporting entity.
        # Newly retrieved docs become visible to the next step, enabling deep multi-hop chains.
        # Stops early on STOP or a repeated/stuck query.
        all_queries = list(queries)
        prior = set(q.lower() for q in queries)
        for _ in range(self.REACT_STEPS):
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
                action = _parse_action(rg.action)
                if action is None or action == "STOP":
                    break
                # avoid repeating an already-issued query (LM is stuck -> stop)
                if action.lower() in prior:
                    break
                self._retrieve_and_fuse(action, self.K_QUERY, ranks, best_rank, passages, guaranteed)
                all_queries.append(action)
                prior.add(action.lower())
            except Exception:
                break

        # ---- Final: guaranteed docs first (by best rank), then fill by RRF score ----
        guaranteed_ordered = sorted(
            guaranteed, key=lambda t: best_rank.get(t, 1_000_000)
        )
        rest = [
            t for t in sorted(ranks, key=lambda t: -ranks[t]) if t not in guaranteed
        ]
        ordered = guaranteed_ordered + rest
        final = [passages[t] for t in ordered[: self.MAX_DOCS] if t in passages]
        return dspy.Prediction(retrieved_docs=final)
