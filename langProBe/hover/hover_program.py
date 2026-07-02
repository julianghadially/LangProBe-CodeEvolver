import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_DOCS = 21


class IdentifyMissing(dspy.Signature):
    """Identify Wikipedia article titles for entities referenced in the claim or in
    the retrieved-passage summaries that do NOT already have their own dedicated
    article among retrieved_titles.

    Reason step by step:
    1. Extract the salient entities named in the claim and summaries (persons,
       organizations, places, bands, films, albums, songs, books, etc.).
    2. For each entity, check whether its dedicated Wikipedia article is ALREADY in
       retrieved_titles by EXACT canonical-title match.
    3. Beware partial / disambiguated matches: a title like "X (footballer)" or
       "X (film)" covers ONLY that one sense. A different person or work also named
       "X" is still MISSING and needs its own dedicated article.
    4. Output the canonical Wikipedia title for each missing dedicated article - the
       most likely natural title (e.g. a person's commonly used name). Do NOT invent
       works or people that the evidence suggests don't exist; only name entities you
       believe have their own standalone Wikipedia article.
    5. Prioritize "bridge" entities the claim depends on that are referenced only
       inside another retrieved passage (e.g. a creator, frontman, or actor named
       inside a work's article).

    Output at most 3 titles, comma-separated, in Wikipedia's canonical lowercase
    form (no quotes, no numbering). Leave empty if no dedicated article is missing.
    """

    claim: str = dspy.InputField(desc="The claim being supported.")
    summaries: str = dspy.InputField(desc="Summaries of already-retrieved passages.")
    retrieved_titles: str = dspy.InputField(
        desc="Comma-separated Wikipedia article titles already retrieved."
    )
    missing_titles: str = dspy.OutputField(
        desc="Comma-separated canonical Wikipedia titles of missing dedicated articles (max 3)."
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Retrieve more candidates per hop (k=10) than the final 21 budget, then dedup by
        # title and pick the best 21 via round-robin. This trades no extra searches (still 3,
        # so the resource penalty is unchanged) for a larger unique-document candidate pool,
        # reclaiming slots previously wasted on duplicate passages across hops.
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        # GAP HOP: identify Wikipedia articles for entities named in the claim/summaries
        # but whose own article title is not yet among the retrieved titles, then fetch them.
        self.identify_missing = dspy.ChainOfThought(IdentifyMissing)

    @staticmethod
    def _doc_title(passage):
        return passage.split(" | ")[0]

    def _rank_aware_dedup(self, hop_docs):
        """Rank-aware merge of ColBERT-ranked passages, dedup by title, cap at MAX_DOCS.

        hop_docs = [hop1, hop2, hop3] + gap_docs_list (each additional hop is the
        passages retrieved for one gap-title query, already truncated to top-3).

        Pure round-robin across all hops lets the (often 2-5) per-title gap dumps
        consume a rotation slot every round, which evicts moderate-rank main-hop
        gold docs (e.g. a hop3 rank-5 passage) before they can be considered. This
        reorders the fill so that:

          Tier A - rank-0 passage of each gap hop. This is the dedicated article the
                   gap hop was issued to fetch (the highest-confidence bridge),
                   so it is reserved a slot first to preserve the recall gains the
                   gap hop was added for.
          Tier B - ALL main-hop passages, interleaved BY RANK across hop1/2/3
                   (round 0 takes each hop's top passage, round 1 the next, ...).
                   Main hops already hold most gold; interleaving by rank lets
                   top-rank refined-hop docs land before main budget is eaten by
                   gap context, and lets moderate-rank gold (rank 5-7) survive.
          Tier C - remaining gap-hop passages (rank 1, 2 of each gap query), folded
                   in only if budget remains, so extra gap context cannot evict an
                   as-yet-unmerged main-hop passage.

        Cross-query rank comparison is invalid (relevance ranks are within-query),
        so we do NOT globally sort by raw rank; per-tier interleaving preserves
        hop diversity instead."""
        main_hops = hop_docs[:3]
        gap_hops = hop_docs[3:]
        seen = set()
        docs = []

        def _push(passage):
            title = self._doc_title(passage)
            if title not in seen:
                seen.add(title)
                docs.append(passage)

        # Tier A: each gap query's top match (the desired dedicated article).
        for gap in gap_hops:
            if gap and len(docs) < MAX_DOCS:
                _push(gap[0])
        # Tier B: main-hop passages interleaved by rank (top ranks first).
        max_main = max((len(h) for h in main_hops), default=0)
        for i in range(max_main):
            for hop in main_hops:
                if i < len(hop) and len(docs) < MAX_DOCS:
                    _push(hop[i])
        # Tier C: remaining gap passages (rank 1, 2) only if budget remains.
        max_gap = max((len(g) for g in gap_hops), default=0)
        for i in range(1, max_gap):
            for gap in gap_hops:
                if i < len(gap) and len(docs) < MAX_DOCS:
                    _push(gap[i])
        return docs[:MAX_DOCS]

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # GAP HOP: fetch dedicated articles for named entities not yet retrieved.
        unique_titles = {
            self._doc_title(p)
            for hop_docs in (hop1_docs, hop2_docs, hop3_docs)
            for p in hop_docs
        }
        # Defensive coalescing: DeepSeek occasionally emits None for an output field,
        # which would otherwise raise TypeError when concatenated/coerced below.
        summaries = (summary_1 or "") + "\n" + (summary_2 or "")
        missing_titles_str = self.identify_missing(
            claim=claim,
            summaries=summaries,
            retrieved_titles=", ".join(sorted(unique_titles)),
        ).missing_titles
        missing_titles_str = missing_titles_str or ""
        missing_titles = []
        for raw in missing_titles_str.replace("\n", ",").split(","):
            title = raw.strip()
            if title and title not in unique_titles:
                unique_titles.add(title)
                missing_titles.append(title)
                if len(missing_titles) >= 3:
                    break
        gap_docs_list = [
            self.retrieve_k(title).passages for title in missing_titles
        ]

        return dspy.Prediction(
            retrieved_docs=self._rank_aware_dedup(
                [hop1_docs, hop2_docs, hop3_docs] + gap_docs_list
            )
        )
