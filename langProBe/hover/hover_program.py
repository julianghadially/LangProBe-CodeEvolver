import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class RerankPassages(dspy.Signature):
    """Listwise re-ranking of retrieved Wikipedia passages for a claim.

    Given a claim, a short summary of the entities/relations that bridge the
    claim, and a numbered list of candidate passages (each rendered as
    "#<id> | <title>"), output the semicolon-separated list of passage IDs in
    order of DECREASING relevance to the claim's supporting-fact retrieval.

    Relevance = how likely the passage is a TOP supporting Wikipedia page that
    directly grounds the entities/relations named in the claim and summary. Put
    the most relevant IDs first. Output ONLY the ordered id list (e.g.
    "3; 7; 1; ...").
    """
    claim: str = dspy.InputField()
    summary: str = dspy.InputField(desc="Key entities and relationships that bridge/support the claim, distilled from retrieved passages.")
    candidates: str = dspy.InputField()
    ranked_ids: str = dspy.OutputField()


class BridgeQuery(dspy.Signature):
    """Generate a Wikipedia search query to find a supporting document that is
    still MISSING for verifying a multi-hop claim.

    The claim connects several entities. The passages already retrieved mention
    other entities that bridge those connections (e.g. a brand advertised in a
    song, a parent company of a product, a place where a road ends). To retrieve
    the missing supporting Wikipedia page, identify the SPECIFIC named entity
    from the retrieved passages that bridges the claim but does not yet have its
    own Wikipedia page in the retrieved set, and build a concise query from that
    entity's name (and a disambiguating word if the name is ambiguous, e.g.
    "Sunkist soft drink", "Microchip Technology company").
    Output only the search query text.
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Titles already retrieved.")
    passages: str = dspy.InputField()
    query: str = dspy.OutputField()


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20
        self.max_docs = 21

        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Bridge-aware query generators using passage content to find missing entities.
        self.create_query_hop2 = dspy.ChainOfThought(BridgeQuery)
        self.create_query_hop3 = dspy.ChainOfThought(BridgeQuery)
        self.create_query_hop4 = dspy.ChainOfThought(BridgeQuery)
        self.create_query_hop5 = dspy.ChainOfThought(BridgeQuery)

        # Keep a short summary focused on entities/connections that bridge the claim.
        self.summarize = dspy.ChainOfThought(
            "claim,passages->summary",
        )

        # Final listwise reranker to pick the most relevant 21 from all candidates.
        self.rerank = dspy.ChainOfThought(RerankPassages)

    def _interleave_dedup(self, hop_doc_lists, max_docs):
        """Round-robin interleave per-hop ranked lists, dedup by title, cap."""
        seen_titles = set()
        merged = []
        max_len = max((len(h) for h in hop_doc_lists), default=0)
        for i in range(max_len):
            for hop_docs in hop_doc_lists:
                if i < len(hop_docs):
                    doc = hop_docs[i]
                    title = doc.split(" | ")[0]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        merged.append(doc)
                        if len(merged) >= max_docs:
                            return merged
        return merged

    def _titles(self, docs):
        return "; ".join(d.split(" | ")[0] for d in docs)

    @staticmethod
    def _norm_q(q):
        # Lowercase, collapse whitespace, and strip punctuation so near-duplicate
        # queries that differ only by punctuation (e.g. "Going Back (2012 film)"
        # vs "Going Back 2012 film") collide and get skipped, forcing later hops
        # onto genuinely different missing entities.
        import re

        return " ".join(re.sub(r"[^\w\s]", " ", (q or "").lower()).split())

    def _pool(self, hop_doc_lists, max_docs):
        """Merge per-hop ranked lists, dedup by title, preserving first-seen
        order (hop1 first, then by rank order within each hop)."""
        seen_titles = set()
        merged = []
        for hop_docs in hop_doc_lists:
            for doc in hop_docs:
                title = doc.split(" | ")[0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    merged.append(doc)
        return merged

    def _render_candidates(self, docs):
        """Render docs as a numbered list '#<id> | <title>'. Truncated preview
        of passage text keeps the prompt small while preserving distinguishability."""
        lines = []
        for i, doc in enumerate(docs):
            title = doc.split(" | ")[0]
            body = doc.split(" | ", 1)[1] if " | " in doc else ""
            preview = body[:400].replace("\n", " ")
            lines.append(f"#{i} | {title} | {preview}")
            if i >= 79:
                break
        return "\n".join(lines)

    def forward(self, claim):
        # HOP 1: retrieve directly with the raw claim.
        used_queries = {self._norm_q(claim)}
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize(claim=claim, passages=hop1_docs).summary

        def _bridge(predictor, prior_docs):
            q = predictor(
                claim=claim,
                retrieved_titles=self._titles(prior_docs),
                passages=prior_docs,
            ).query
            nq = self._norm_q(q)
            if not q or nq in used_queries:
                return []
            used_queries.add(nq)
            return self.retrieve_k(q).passages

        # HOP 2: derive a bridge query targeting an entity named in the
        # retrieved passages but not yet retrieved as its own page.
        hop2_docs = _bridge(self.create_query_hop2, hop1_docs)
        summary_2 = self.summarize(
            claim=claim, passages=hop1_docs + hop2_docs
        ).summary

        # HOP 3: bridge query after considering hop1 + hop2 retrieved titles.
        hop3_docs = _bridge(self.create_query_hop3, hop1_docs + hop2_docs)

        # HOP 4: a final bridge query targeting any entity still missing after
        # the first three retrievals (uses all passages gathered so far).
        all_prior_docs = hop1_docs + hop2_docs + hop3_docs
        hop4_docs = _bridge(self.create_query_hop4, all_prior_docs)

        # HOP 5: extra bridge query if entities still appear missing.
        all_prior_docs_4 = hop1_docs + hop2_docs + hop3_docs + hop4_docs
        hop5_docs = _bridge(self.create_query_hop5, all_prior_docs_4)

        # Build the full candidate pool (dedup, order hop1 first then by rank).
        pool = self._pool(
            [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs],
            self.max_docs,
        )

        # Single-hop fallback ordering for documents the reranker omits.
        interleave_fallback = self._interleave_dedup(
            [hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs],
            len(pool),
        )

        # Distilled gap analysis over all gathered passages: a concise statement
        # of the entities/relations that bridge the claim. Feeding it to the
        # listwise reranker grounds relevance on the claim's support structure,
        # helping borderline supporting pages rank above look-alikes.
        rerank_summary = self.summarize(
            claim=claim, passages=interleave_fallback[:40]
        ).summary

        reranked = self._rerank_to_max(pool, claim, rerank_summary, interleave_fallback)
        return dspy.Prediction(retrieved_docs=reranked)

    def _rerank_to_max(self, pool, claim, summary, fallback):
        """Use the listwise reranker to order the pool then cap at max_docs.
        Any pool docs the LM omits are appended in fallback order so the
        returned count is exactly max_docs."""
        if len(pool) <= self.max_docs:
            return pool[: self.max_docs]
        candidates = self._render_candidates(pool)
        ranked_ids_raw = self.rerank(
            claim=claim, summary=summary, candidates=candidates
        ).ranked_ids or ""
        ordered_ids = []
        seen = set()
        for tok in ranked_ids_raw.replace(",", ";").split(";"):
            tok = tok.strip().lstrip("#")
            if tok.isdigit():
                i = int(tok)
                if 0 <= i < len(pool) and i not in seen:
                    seen.add(i)
                    ordered_ids.append(i)
        ordered = [pool[i] for i in ordered_ids]
        # Append fallback docs the LM did not select, to reach max_docs.
        for doc in fallback:
            title = doc.split(" | ")[0]
            if doc in ordered:
                continue
            elif any(d.split(" | ")[0] == title for d in ordered):
                continue
            ordered.append(doc)
            if len(ordered) >= self.max_docs:
                break
        return ordered[: self.max_docs]