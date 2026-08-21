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


class Expand(dspy.Signature):
    """You are a Wikipedia retrieval expert. You have a factual claim, a summary of
    Wikipedia passages already retrieved, and the search queries already used.

    Write NEW follow-up search queries targeting entities, people, organizations,
    places, or works DISCOVERED in the summary that are needed to verify the claim but
    NOT already covered by the prior queries (these are the multi-hop connections).

    Rules:
    - Only target entities/aspects NOT already in prior_queries.
    - Use concise keyword queries (entity name + short descriptor), NOT full sentences.
    - Write 2 to 3 distinct queries, exactly one per line."""

    claim = dspy.InputField()
    summary = dspy.InputField()
    prior_queries = dspy.InputField()
    search_queries = dspy.OutputField(
        desc="2 to 3 NEW follow-up Wikipedia search queries, one per line"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval for a claim, optimized for document recall.

    Pipeline:
      1. Decompose the claim into several diverse search queries (one per entity/aspect).
      2. Retrieve a large candidate set per query (high k) and fuse across queries with
         Reciprocal Rank Fusion (RRF), deduplicating by document title.
      3. Summarize the top fused passages and generate multi-hop follow-up queries for
         entities discovered but not yet searched, then retrieve and fuse those too.
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

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=self.K_QUERY)
        self.decompose = dspy.ChainOfThought(Decompose)
        self.summarize = dspy.ChainOfThought("claim,passages->summary")
        self.expand = dspy.ChainOfThought(Expand)

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

        # ---- Stage 2: summarize top passages, then multi-hop expansion ----
        try:
            top_titles = sorted(ranks, key=lambda t: -ranks[t])[:15]
            top_passages = [passages[t] for t in top_titles if t in passages]
            summary = self.summarize(claim=claim, passages=top_passages).summary
            exp = self.expand(
                claim=claim,
                summary=summary,
                prior_queries="\n".join(queries),
            ).search_queries
            follow_ups = _dedupe_queries(_split_queries(exp))
            prior = set(queries)
            follow_ups = [q for q in follow_ups if q not in prior][:3]
            for q in follow_ups:
                self._retrieve_and_fuse(q, self.K_QUERY, ranks, best_rank, passages, guaranteed)
        except Exception:
            pass

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
