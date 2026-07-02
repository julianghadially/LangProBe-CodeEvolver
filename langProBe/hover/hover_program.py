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
        """Round-robin merge that consolidates the gap retrievals into ONE pool.

        hop_docs = [hop1, hop2, hop3] + gap_docs_list, where each gap hop is the
        passages retrieved for one gap-title query (already truncated to top-3).

        Why consolidate: the previous pure round-robin passed each per-title gap
        retrieval as its OWN hop, so N gap titles gave the gap hop N votes per
        rotation. With the iter-3 gap-hop cap of 3 titles, that meant 6 hops total
        and gap took ~half of the 21 slots, evicting moderate-rank main-hop gold
        (e.g. a hop3 rank-5 passage) before it could be considered, and -(when a
        gap title was a wrong "disambiguation-style" guess)- letting 3 wrong-gap
        ranks evict an additional main gold.

        We instead merge all gap retrievals into a single rank-interleaved pool
        (round 0 of the pool takes each gap title's rank-0 passage, etc.), so the
        gap hop contributes exactly ONE vote per round-robin rotation. The
        round-robin then runs across [hop1, hop2, hop3, gap_pool]: 4 hops, so main
        gets ~3/4 of the 21 slots (rank 0-5 of each main hop survives) and the gap
        pool gets ~1/4 (still surfaces every gap-rank-0 bridge candidate first).
        Same ColBERT rank ordering and dedup-by-title as before. No structural
        change for the no-gap case (gap_pool empty -> old 3-hop interleave)."""
        main_hops = hop_docs[:3]
        gap_hops = hop_docs[3:]
        # Build ONE rank-interleaved pool across all gap-title retrievals:
        # rank-0 of every gap title first, then rank-1 of every title, etc.
        gap_pool = []
        max_gap = max((len(g) for g in gap_hops), default=0)
        for i in range(max_gap):
            for gap in gap_hops:
                if i < len(gap):
                    gap_pool.append(gap[i])
        merge_hops = main_hops + ([gap_pool] if gap_pool else [])

        seen = set()
        docs = []
        max_len = max((len(h) for h in merge_hops), default=0)
        for i in range(max_len):
            for hop in merge_hops:
                if i < len(hop):
                    title = self._doc_title(hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(hop[i])
                        if len(docs) >= MAX_DOCS:
                            return docs
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
