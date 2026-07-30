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


class BridgeQueries(dspy.Signature):
    """Generate several DIVERSE Wikipedia search queries — one per DIFFERENT
    still-missing supporting page — to retrieve pages for verifying a multi-hop
    claim.

    A multi-hop claim has ~3 supporting Wikipedia pages. retrieved_titles
    already holds some of these (or related pages). Each query you output MUST
    target a DIFFERENT still-missing supporting page. Use BOTH strategies across
    your outputs:

    A) CROSS-REF TITLE — scan the passages' text for a Wikipedia page title that
    is *referenced* inside a retrieved passage (a work title in quotes/italics, a
    person's full name, a co-star/colleague name, a company, a genus, a
    linked-style entity) whose own page is NOT in retrieved_titles. Use that
    EXACT title verbatim. CRITICAL: do NOT only look for titles that "make the
    claim true". The retrieval task needs EVERY supporting page, and many claims
    are FALSE — the missing page is often a titled entity printed right in the
    passages that actually CONTRADICTS the claim's wording (e.g. an actor's other
    film listed in their filmography, a real co-star whose name differs from the
    one the claim asserts, the actual company behind a product, the genus above a
    binomial). Harvest those exact verbatim titles even when they disprove the
    claim, and query them as-is.

    B) COMBINED-ENTITY QUERY — build a query that JOINS TWO key named entities
    (prefer one entity from the claim + one entity from the retrieved passages
    that completes the chain), separated by a single concise domain term likely
    to appear on the missing page, e.g.
        "Charles Bronson Leslie Nielsen comedy director"
        "The Secret Agent Stephen Graham"
        "New York Islanders 1974 75 season"
        "Dieffenbachia Carlina flowering plant genus"
    A combined query surfaces the page that BRIDGES two entities, which
    single-entity queries routinely miss.

    Rules:
    - Each query must target a DIFFERENT missing supporting entity/page.
    - Do NOT output a single entity name already present in retrieved_titles.
    - Do NOT echo the whole claim and do NOT repeat entities/queries.
    - If a claim entity is misspelled or a garbled quote, correct it using the
      plausible spelling guided by the passages.
    - Prefer named-entity / title cross-references over generic descriptor
      phrases; include at most one generic descriptor-style combined query.
    - Output a semicolon-separated list of queries, e.g.
      "q1 ; q2 ; q3". Output ONLY that list.
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Titles already retrieved.")
    passages: str = dspy.InputField()
    queries: str = dspy.OutputField(desc="Semicolon-separated list of diverse search queries.")


class RefTitles(dspy.Signature):
    """Harvest VERBATIM Wikipedia article titles that are *referenced* inside
    the supplied passages but NOT yet in retrieved_titles. A multi-hop claim
    has ~3 supporting pages; frequently a retrieved passage names the
    still-missing supporting page right in its text (a person's co-star, a work
    title in quotes/italics, a company, a genus, a disambiguated film/work).
    Often the missing page even CONTRADICTS the claim's wording (e.g. an actor's
    other film listed in their filmography, the actual co-star whose name
    differs from the one asserted in the claim, the company behind a product,
    the genus above a binomial). Harvest those EXACT verbatim titles REGARDLESS
    of whether they confirm the claim — the retrieval task needs EVERY
    supporting page.

    PRIORITY — scan the FIRST SENTENCE of every passage for the entity that
    FILLS THE CLAIM'S GAP. A claim describes a missing subject indirectly
    ("the company that made X", "this religion", "the director of Y",
    "the choreographer born in YYYY", "the genus of ..."). The passage then
    NAMES that subject verbatim in its lead sentence (e.g. claim "the company
    that made Welcome to Macintosh" -> passage says "...focusing on computer
    company Apple Inc."; claim "In this religion, where Mehet-Weret..." ->
    passage says "a goddess of the sky in Ancient Egyptian religion"). ALWAYS
    output that named gap-filling entity first, in its exact printed form
    ("Apple Inc.", "Ancient Egyptian religion", "Gene Kelly",
    "Best Foot Forward (musical)"), as a standalone title.

    Rules:
    - Output ONLY exact verbatim titles copied from the passage text (use the
      disambiguated form printed in the text, e.g. 'This Is England (film)',
      'Best Foot Forward (musical)', 'Secret Agent (TV series)',
      'Josh Flitter', 'Gene Kelly', 'Airlines of Africa', 'Warren Fu',
      'Apple Inc.', 'Ancient Egyptian religion').
    - Do NOT invent or paraphrase; do NOT include a title already in
      retrieved_titles; do NOT include the page's own title.
    - Prefer the GAP-FILLING subject of the claim over obscure tangential
      cross-references; then add other bridging titles (co-stars, directors,
      filmography entries, parent/child taxa, related companies/works).
    - Do NOT skip a title because it is a large/famous entity (Apple Inc.,
      Gene Kelly) — if it is named in a passage and not yet retrieved, output
      it.
    - Output a semicolon-separated list of at most 6 titles, e.g.
      "Apple Inc. ; Ancient Egyptian religion ; Gene Kelly ; ...". Output ONLY
      that list (or empty if no referenced-but-missing title is present).
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Titles already retrieved (skip these).")
    passages: str = dspy.InputField()
    titles: str = dspy.OutputField(desc="Semicolon-separated list of verbatim referenced-but-missing Wikipedia titles.")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25
        self.max_docs = 21
        self.batch_queries = 5
        self.batch_queries_2 = 2
        self.ref_titles = 4
        self.ref_titles_early = 4

        self.retrieve_k = dspy.Retrieve(k=self.k)

# Small-k retriever used by the verbatim-cross-ref-title batch: a near-exact
        # title query only needs a couple of top hits, so we avoid flooding the
        # pool (which would crowd out supporting pages in the reranker). k=2
        # guards against the exact-title page not being colbert's #1 hit.
        self.retrieve_title_k = dspy.Retrieve(k=2)

        # One LM call generates several DIVERSE bridge queries at once, each
        # targeting a DIFFERENT missing supporting page (cross-ref titles and
        # combined-entity queries). Far higher recall than one query per hop.
        self.bridge_queries = dspy.ChainOfThought(BridgeQueries)

        # Optional second refinement batch: more diverse bridge queries after
        # seeing the first batch's retrieved titles/passages.
        self.bridge_queries_2 = dspy.ChainOfThought(BridgeQueries)

# Dedicated cross-ref title harvester: pulls EXACT verbatim Wikipedia titles
        # referenced inside already-retrieved passages that aren't yet retrieved,
        # then queries those titles directly. Recovers supporting pages the
        # bridge queries miss because the LM latched onto a wrong bridging path.
        self.ref_titles_lm = dspy.ChainOfThought(RefTitles)

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

    def _parse_queries(self, raw):
        """Parse a semicolon-separated (or newline) list of queries."""
        out = []
        if not raw:
            return out
        for part in raw.replace(",", ";").replace("\n", ";").split(";"):
            q = part.strip().strip('"').strip("'").strip()
            # Drop leading list markers like "1)" / "-".
            q = q.lstrip("-").strip()
            if " " in q and q[:2].rstrip(")").isdigit():
                q = q.split(None, 1)[1].strip()
            if q:
                out.append(q)
        return out

    def _run_bridge_batch(self, predictor, claim, prior_docs, used_queries, hop_lists, limit):
        """Generate a batch of diverse bridge queries and retrieve each (after
        dedup). Append each retrieval result to hop_lists in place. Returns
        updated prior_docs."""
        # Cap passage payload size to keep the prompt manageable.
        passages = prior_docs[-100:] if len(prior_docs) > 100 else prior_docs
        res = predictor(
            claim=claim,
            retrieved_titles=self._titles(prior_docs),
            passages=passages,
        )
        queries = self._parse_queries(res.queries)[:limit]
        for q in queries:
            nq = self._norm_q(q)
            if not q or nq in used_queries:
                continue
            used_queries.add(nq)
            docs = self.retrieve_k(q).passages
            hop_lists.append(docs)
            prior_docs = prior_docs + docs
        return prior_docs

    def _render_passage_previews(self, docs, preview_chars=260, cap=200):
        """Render docs as 'TITLE: <first preview_chars of body>' so the
        harvester can scan MANY passages (and their opening cross-references)
        within a modest prompt budget, instead of a few full passages that it
        ends up skimming. Wikipedia cross-reference titles almost always appear
        in the lead sentence, so a short preview preserves them while letting
        the LM see far more candidate passages at once."""
        lines = []
        for doc in docs[-cap:]:
            parts = doc.split(" | ", 1)
            title = parts[0]
            body = parts[1] if len(parts) > 1 else ""
            preview = body[:preview_chars].replace("\n", " ")
            lines.append(f"{title}: {preview}")
        return "\n".join(lines)

    def _run_ref_titles_batch(self, claim, prior_docs, used_queries, hop_lists, limit):
        """Harvest verbatim referenced-but-missing titles from retrieved
        passages and query each as-is (verbatim). Dedup against already-used
        queries/titles."""
        passages = self._render_passage_previews(prior_docs)
        res = self.ref_titles_lm(
            claim=claim,
            retrieved_titles=self._titles(prior_docs),
            passages=passages,
        )
        titles = self._parse_queries(res.titles)[:limit]
        retrieved_titles = {self._norm_q(d.split(" | ")[0]) for d in prior_docs}
        for t in titles:
            nq = self._norm_q(t)
            if not t or nq in used_queries or nq in retrieved_titles:
                continue
            used_queries.add(nq)
            docs = self.retrieve_title_k(t).passages
            hop_lists.append(docs)
            prior_docs = prior_docs + docs
        return prior_docs

    def forward(self, claim):
        # HOP 1: retrieve directly with the raw claim.
        used_queries = {self._norm_q(claim)}
        hop_lists = []
        hop1_docs = self.retrieve_k(claim).passages
        hop_lists.append(hop1_docs)

        # BATCH 0 (early): harvest verbatim cross-ref titles straight from the
        # high-signal hop1 passages and retrieve them BEFORE the bridge batches
        # run. Hop1 is a pure colbert match for the claim and frequently names
        # the still-missing supporting pages right in its text; harvesting them
        # early (a) places those supporting pages in the pool before the bridge
        # LM can latch onto a wrong bridging path, and (b) gives the bridge
        # batches richer context (the harvested titles' passages) to build on.
        prior_docs = list(hop1_docs)
        prior_docs = self._run_ref_titles_batch(
            claim, prior_docs, used_queries, hop_lists, self.ref_titles_early,
        )

        # BATCH 1: several diverse bridge queries from claim + hop1 passages.
        prior_docs = self._run_bridge_batch(
            self.bridge_queries, claim, prior_docs, used_queries, hop_lists,
            self.batch_queries,
        )

        # BATCH 2: a small refinement batch after seeing batch-1 retrievals.
        summary_2 = self.summarize(claim=claim, passages=prior_docs).summary  # noqa: F841
        prior_docs = self._run_bridge_batch(
            self.bridge_queries_2, claim, prior_docs, used_queries, hop_lists,
            self.batch_queries_2,
        )

# BATCH 3: harvest verbatim cross-ref titles from all retrieved passages
        # and query them as-is. Recovers supporting pages the bridge queries miss.
        prior_docs = self._run_ref_titles_batch(
            claim, prior_docs, used_queries, hop_lists, self.ref_titles,
        )

        # Build the full candidate pool (dedup, order hop1 first then by rank).
        pool = self._pool(hop_lists, self.max_docs)

        # Single-hop fallback ordering for documents the reranker omits.
        interleave_fallback = self._interleave_dedup(hop_lists, len(pool))

        # Distilled gap analysis over all gathered passages.
        rerank_summary = self.summarize(
            claim=claim, passages=interleave_fallback[:40]
        ).summary

        reranked = self._rerank_to_max(
            pool, claim, rerank_summary, interleave_fallback, hop_lists,
            anchor_norms=used_queries,
        )
        return dspy.Prediction(retrieved_docs=reranked)

    def _inject_anchors(self, ordered, pool, anchor_norms):
        """Guarantee that 'anchor' candidate pages already in the pool end up
        in the final selection.

        An anchor is a candidate whose normalized title exactly equals a
        previously-issued query. Such matches are overwhelmingly verbatim
        Wikipedia titles (cross-ref titles harvested from passages, or single
        title-style bridge queries); the listwise reranker occasionally drops
        them in favour of noisier topically-similar pages even though they are
        actually supporting pages, which zeros out the whole example. We force
        any missing anchors into the selection by swapping out the
        lowest-priority non-anchor slots — reranker / fallback order is
        otherwise untouched."""
        if not anchor_norms:
            return ordered

        def norm(doc):
            return self._norm_q(doc.split(" | ")[0])

        present_norms = set(norm(d) for d in ordered)
        missing = [
            d for d in pool
            if norm(d) in anchor_norms and norm(d) not in present_norms
        ]
        if not missing:
            return ordered

        # Only consider anchors actually retrievable from the pool; preserve
        # pool ordering so the first (most relevant) occurrence is kept.
        seen_anchor_norms = set()
        unique_missing = []
        for d in missing:
            nt = norm(d)
            if nt not in seen_anchor_norms:
                seen_anchor_norms.add(nt)
                unique_missing.append(d)

        result = list(ordered)
        protected = set(
            norm(d) for d in result if norm(d) in anchor_norms
        )
        for m in unique_missing:
            # Replace the last non-anchor slot with this anchor.
            target = None
            for i in range(len(result) - 1, -1, -1):
                nt = norm(result[i])
                if nt not in anchor_norms and nt not in protected:
                    target = i
                    break
            if target is None:
                break
            result[target] = m
            protected.add(norm(m))
        return result

    def _rerank_to_max(self, pool, claim, summary, fallback, hop_lists=None,
                      anchor_norms=None):
        """Use the listwise reranker to order the pool then cap at max_docs.
        Any pool docs the LM omits are appended in fallback order so the
        returned count is exactly max_docs. Anchor pages (titles that exactly
        match an issued query) are guaranteed inclusion even when the
        reranker omits them."""
        if anchor_norms is None:
            anchor_norms = set()
        if len(pool) <= self.max_docs:
            return self._inject_anchors(pool[: self.max_docs], pool, anchor_norms)
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
        ordered = ordered[: self.max_docs]
        return self._inject_anchors(ordered, pool, anchor_norms)