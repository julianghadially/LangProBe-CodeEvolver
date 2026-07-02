import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_DOCS = 21

MAIN_RESERVE = 15
MAX_GAP_TITLES = 3
GAP_PASSAGES_PER_TITLE = 3
PASSAGE_SNIPPET_CHARS = 600
SNIPPET_DOCS_PER_HOP = 5
MAX_GAP_ITERATIONS = 2


class IdentifyMissing(dspy.Signature):
    """Identify Wikipedia articles that are needed to verify the claim but are
    NOT yet present in the retrieved document set. You are one ITERATIVE step in a
    multi-hop retrieval procedure: in earlier rounds other articles may already
    have been retrieved on your suggestion — your job this round is to look one
    hop FURTHER and name any bridge articles still missing.

    Inputs:
      - claim: the factual claim we are fact-checking.
      - retrieved_titles: the canonical Wikipedia article titles already retrieved
        (NONE of these should be re-suggested). Includes BOTH main-hop titles and
        any titles already retrieved by prior gap rounds.
      - previously_suggested_titles: titles you named in PRIOR gap rounds. NEVER
        repeat or re-suggest any title already in this list — repeating wastes a
        retrieval slot. Aim only for genuinely NEW bridges.
      - passage_snippets: LITERAL truncated text excerpts (title + opening body)
        from a sample of retrieved articles — INCLUDING articles retrieved by
        prior gap rounds. These expanded snippets may surface deeper-bridge
        entity mentions that the original main-hop snippets lacked. Match entity
        MENTIONS by text span, not gist.

    Reasoning steps:
      1. Enumerate every distinct entity named in the claim (people, orgs,
         works/titles, places, dates-as-eras). For each, decide whether its
         dedicated standalone article is already in `retrieved_titles`. Match by
         EXACT canonical title only — a disambiguated near-name such as
         "Dave Evans (singer)" covers ONLY that sense; do not assume "Dave Evans"
         alone is the same article.
      2. Scan `passage_snippets` for any named entity MENTIONED inside a passage
         but whose own standalone article is NOT in `retrieved_titles`. Voice
         your entity extraction aloud, citing the snippet fragments where the
         mention occurs. The most common supporting-fact miss is a "bridge"
         entity that appears only by surname or partial name inside another
         retrieved doc (e.g. a band article that opens "frontman Billy Corgan").
         Surface these mentions even when the claim only refers to them indirectly.
      3. LOOK ONE HOP FURTHER THAN THE LAST ROUND: prior gap rounds retrieved
         fresh articles. Open and read those new snippets: they may NAME a
         further, more obscure bridge entity (the article a prior-round title is
         the father/teacher/label/source-work OF) whose own article is still
         missing. A bridge that is two entity-mentions deep (mention mentions
         mention-2 which mentions article-3) typically only surfaces once
         article-2 has been retrieved and its snippet opened — surface such
         two-deep bridges this round.
      4. Also extract works/roles/items NAMED WITHIN passage snippets —
         filmography/discography entries, cast lists, "remake of"/spin-off/source
         works, season rosters, "directed by", "fronted by", "founded by".
      5. SPECIFIC OVER GENERIC: the claim's "bridge" entity is usually the
         SPECIFIC named work/show/episode/role/org, not the generic
         network/channel/studio/publisher/platform that aired or released it.
         Prefer the specific named thing even when a snippet only names the
         channel. Do NOT add a network/platform/studio title unless it is itself
         the claimed bridge. EXCEPTION: when the claim itself asks about a broad
         TOPIC or subject area (e.g. "X religion", "X education", "X history"),
         the bridge is often the broad overview/concept article on that topic —
         prefer it over enumerating specific sub-entities (deities, schools,
         events) that the topic article would cover.
      6. EVIDENCE-BOUND — NO OUTSIDE KNOWLEDGE: name ONLY entities that appear
         LITERALLY in a passage snippet OR are named verbatim in the claim. Do
         NOT use your outside knowledge to identify what a retrieved article is
         about if its snippet was not provided — if a retrieved title has no
         snippet here, you cannot know its contents, so do NOT guess its cast,
         members, or subjects from memory. Inventing a plausible-sounding entity
         from memory and naming its article is the most common wrong-bridge
         failure mode.
      7. COMPLETENESS OVER SALIENCE: enumerate EVERY entity MENTIONED in a
         snippet that plausibly relates to the claim and has a standalone
         article, not just the most prominent one. Supporting-fact golds are
         frequently the LESS salient participant — the runner-up (not the
         winner), the supporting cast member, the secondary band, the item
         listed last in a roster. If a snippet names multiple participants in the
         claim's event/category, list each one whose own article is missing; do
         not silently dismiss the less-prominent ones as "not directly relevant".
      8. THE PERSON IS OFTEN THE BRIDGE: if the claim names a person (director,
         author, actor, founder, performer), consider whether that PERSON's own
         standalone article is missing — the person's own article is frequently
         the supporting fact, not just their works. Do NOT assume retrieving only
         their films/works/roles suffices; if the person article itself is not in
         `retrieved_titles`, suggest it.
      9. PRECISION FIRST, then COMPLETENESS: only suggest titles you are
         reasonably confident exist as standalone Wikipedia articles. Listing
         multiple plausible candidates (alternate name-forms, different-sense
         disambiguations) is BETTER than one exact guess — but never INVENT a
         title that is not a real Wikipedia article, and never echo one in
         `retrieved_titles` or `previously_suggested_titles`.
10. STOP EARLY: if you cannot identify any genuinely new missing bridge that
           exists as a standalone article, output an empty string. Do not pad with
           marginal or already-suggested titles.

    Output:
      - missing_titles: up to 3 NEW canonical Wikipedia article titles to
        retrieve next, ordered by how likely each is the claim's missing bridge.
        EMIT EXACTLY ONE TITLE PER LINE (never comma-separated). Many Wikipedia
        titles themselves contain commas (e.g. "Murray Hill, Manhattan",
        "Washington, D.C.", "Skittles, Mars, Incorporated"), so
        comma-separation is ambiguous and will be mis-parsed — use one title per
        line only. None may appear in `retrieved_titles` or
        `previously_suggested_titles`. Empty if no genuinely new bridge remains.
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Canonical Wikipedia titles already retrieved (main hops + prior gap rounds), one per line. NEVER re-suggest these.")
    previously_suggested_titles: str = dspy.InputField(desc="Canonical Wikipedia titles you named in prior gap rounds, one per line. NEVER repeat these.")
    passage_snippets: str = dspy.InputField(desc="Literal truncated text excerpts from a sample of retrieved articles (including prior gap rounds). Scan these for entity MENTIONS by text span.")

    missing_titles: str = dspy.OutputField(desc="Up to 3 NEW canonical Wikipedia titles to retrieve next, ONE TITLE PER LINE (never comma-separated — titles may themselves contain commas). None already in retrieved_titles or previously_suggested_titles. Empty if no genuinely new bridge remains.")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - This system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.identify_missing = dspy.ChainOfThought(IdentifyMissing)

    # Defensive LM-call wrapper. DeepSeek-V4-Flash occasionally emits a refusal
    # (Chinese "你好..." / "无法..." or an English "I cannot") instead of the
    # structured payload dspy's adapter expects, raising AdapterParseError.
    # For the previously-UNGUARDED main-hop call sites below that propagated out
    # of `forward` and the whole pipeline scored a free ZERO. We retry once (a
    # redraw usually avoids the refusal); on a second failure we degrade to a
    # caller-supplied default so the pipeline keeps running. We deliberately
    # do NOT screen the *text* of valid completions — only catch real parse
    # exceptions — to avoid false positives on legitimate prose summaries.
    def _lm_call(self, predictor, field, default, **kwargs):
        for attempt in range(2):
            try:
                resp = predictor(**kwargs)
                value = getattr(resp, field, None)
                if value is None:
                    value = ""
                return str(value)
            except Exception:
                pass  # retry / degrade
        return default

    @staticmethod
    def _doc_title(passage):
        return passage.split(" | ")[0]

    def _round_robin_dedup(self, hop_docs):
        """Round-robin across hops (each ColBERT-ranked), dedup by title, cap at MAX_DOCS."""
        seen = set()
        docs = []
        max_len = max((len(h) for h in hop_docs), default=0)
        for i in range(max_len):
            for hop in hop_docs:
                if i < len(hop):
                    title = self._doc_title(hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(hop[i])
                        if len(docs) >= MAX_DOCS:
                            return docs
        return docs[:MAX_DOCS]

    def _main_first_merge(self, main_hops, gap_pool_hop_by_iter, gap_hops_flat):
        """Merge that protects main-hop gold from gap-hop displacement.

        Phase 1 reserves the first ``MAIN_RESERVE`` of the cap for the main hops only
        (round-robin across main hops in ColBERT rank order), so a rank-5-7 main-hop
        gold document can never be evicted by a low-rank gap dump. Phase 2 fills the
        remaining tail slots from the rank-interleaved gap pool (separated round by
        round so earlier, higher-confidence bridges are preferred). Phase 3 (only
        reached if the gap pool was too small to fill the cap) resumes the main-hop
        round-robin for whatever ranks were skipped, so a small gap pool never
        wastes budget."""
        seen = set()
        docs = []
        max_main = max((len(h) for h in main_hops), default=0)

        # Phase 1: reserve the first MAIN_RESERVE slots for main hops only.
        for i in range(max_main):
            for hop in main_hops:
                if i < len(hop):
                    title = self._doc_title(hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(hop[i])
                        if len(docs) >= MAIN_RESERVE:
                            break
            if len(docs) >= MAIN_RESERVE:
                break

        # Phase 2: tail slots from a rank-interleaved gap pool.
        # Pool each round's per-title retrievals; interleave across titles so no
        # single wrong bridge dominates the tail. Earlier iterations (higher
        # confidence, since IdentifyMissing already saw main-hop evidence for
        # them first) come before later iterations.
        max_titles = max((len(h) for h in gap_pool_hop_by_iter), default=0) \
            if gap_pool_hop_by_iter else 0
        for i in range(max_titles):
            for iter_hop in gap_pool_hop_by_iter:
                if i < len(iter_hop):
                    title = self._doc_title(iter_hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(iter_hop[i])
                        if len(docs) >= MAX_DOCS:
                            return docs

        # Fallback into any remaining gap docs not picked by the round interleaver.
        if gap_hops_flat:
            for passage in gap_hops_flat:
                title = self._doc_title(passage)
                if title not in seen:
                    seen.add(title)
                    docs.append(passage)
                    if len(docs) >= MAX_DOCS:
                        return docs

        # Phase 3: if gap pool was small and there is leftover budget, resume
        # the main-hop round-robin for ranks not yet taken.
        for i in range(max_main):
            for hop in main_hops:
                if i < len(hop):
                    title = self._doc_title(hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(hop[i])
                        if len(docs) >= MAX_DOCS:
                            return docs
        return docs[:MAX_DOCS]

    @staticmethod
    def _build_snippets(hop_docs_list, docs_per_hop=SNIPPET_DOCS_PER_HOP,
                        snippet_chars=PASSAGE_SNIPPET_CHARS):
        """Compose literal title+opening-text snippets from the top passages of each hop.

        Exposing the raw surface text (not LM summaries) lets IdentifyMissing match
        entity MENTIONS by text span — the dominant miss mode (a bridge entity named
        only by surname/partial-name inside another retrieved article)."""
        snippets = []
        for hop_docs in hop_docs_list:
            for passage in hop_docs[:docs_per_hop]:
                if " | " in passage:
                    title, body = passage.split(" | ", 1)
                else:
                    title, body = passage, ""
                title = title.strip()
                body = body.strip()
                if len(body) > snippet_chars:
                    body = body[:snippet_chars] + "…"
                snippets.append(f"{title} :: {body}")
        return "\n".join(snippets)

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self._lm_call(
            self.summarize1, "summary", "",
            claim=claim, passages=hop1_docs,
        )

        # HOP 2
        hop2_query = self._lm_call(
            self.create_query_hop2, "query", claim,
            claim=claim, summary_1=summary_1,
        )
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self._lm_call(
            self.summarize2, "summary", "",
            claim=claim, context=summary_1, passages=hop2_docs,
        )

        # HOP 3
        hop3_query = self._lm_call(
            self.create_query_hop3, "query", hop2_query or claim,
            claim=claim, summary_1=summary_1, summary_2=summary_2,
        )
        hop3_docs = self.retrieve_k(hop3_query).passages

        main_hops = [hop1_docs, hop2_docs, hop3_docs]

        # Build the retrieved-title set seeded with main-hop titles.
        retrieved_titles = []
        seen_titles = set()
        for hop_docs in main_hops:
            for passage in hop_docs:
                title = self._doc_title(passage)
                if title not in seen_titles:
                    seen_titles.add(title)
                    retrieved_titles.append(title)

        # ADAPTIVE GAP-RETRIEVAL LOOP — up to MAX_GAP_ITERATIONS rounds. Each
        # round lets IdentifyMissing re-scan the EXPANDED snippet pool (main
        # hops + gap docs retrieved by prior rounds) so deeper multi-hop
        # bridges whose surface mention only opens IN a prior round's freshly
        # retrieved bridge article can surface. Stops early once IdentifyMissing
        # emits no genuinely new title.
        gap_hops_by_iter = []   # list (per round) of per-title retrieval slices
        gap_pool_by_iter = []   # rank-interleaved within a round for the merge
        gap_hops_flat = []      # concatenation of all gap passages (fallback + snippets)
        previously_suggested = []  # LM-suggested titles across all prior rounds
        previously_seen = set()

        for _ in range(MAX_GAP_ITERATIONS):
            # Snippets across main hops + gap docs retrieved so far.
            hop_docs_for_snippets = list(main_hops) + [
                sl for sl in gap_hops_by_iter for sl in sl
            ]
            snippets = self._build_snippets(hop_docs_for_snippets)

            titles_str = "\n".join(retrieved_titles) if retrieved_titles else ""
            prev_str = "\n".join(previously_suggested) if previously_suggested else ""

            missing_titles_raw = self._lm_call(
                self.identify_missing, "missing_titles", "",
                claim=claim,
                retrieved_titles=titles_str,
                previously_suggested_titles=prev_str,
                passage_snippets=snippets,
            ).strip()

            # Parse candidate titles split ONE PER LINE (the prompt instructs the
            # LM to emit exactly one title per line, because Wikipedia titles
            # themselves may contain commas — splitting on commas corrupts
            # titles like "Murray Hill, Manhattan" / "Skittles, Mars,
            # Incorporated" into garbage queries. Fall back to newline/semicolon
            # tokenisation only.)
            new_titles = []
            for raw in [t.strip() for t in missing_titles_raw.replace(";", "\n").splitlines()]:
                if not raw or raw in seen_titles or raw in previously_seen:
                    continue
                previously_seen.add(raw)
                new_titles.append(raw)
                if len(new_titles) >= MAX_GAP_TITLES:
                    break

            if not new_titles:
                break  # IdentifyMissing considers the evidence complete.

            previously_suggested.extend(new_titles)

            # Retrieve each new title verbatim; record per-title top-N slices
            # AND a rank-interleaved pool for this round's contribution to the
            # main-first merge.
            per_title_slices = []  # [ [passage, ...], [passage, ...], ... ]
            for raw in new_titles:
                try:
                    gap_docs = self.retrieve_k(raw).passages
                except Exception:
                    gap_docs = []
                slice_docs = gap_docs[:GAP_PASSAGES_PER_TITLE]
                per_title_slices.append(slice_docs)
                gap_hops_flat.extend(slice_docs)
                # Add the retrieved titles (from the top of each slice) to the
                # running retrieved set so the NEXT round and the seen_titles
                # guard reflect expansion.
                for passage in gap_docs:
                    title = self._doc_title(passage)
                    if title not in seen_titles:
                        seen_titles.add(title)
                        retrieved_titles.append(title)

            gap_hops_by_iter.append(per_title_slices)

            # Rank-interleaved pool for this round: round-robin across titles
            # by ColBERT rank.
            round_pool = []
            max_len = max((len(s) for s in per_title_slices), default=0)
            for i in range(max_len):
                for s in per_title_slices:
                    if i < len(s):
                        round_pool.append(s[i])
            gap_pool_by_iter.append(round_pool)

        if gap_pool_by_iter:
            retrieved_docs = self._main_first_merge(
                main_hops, gap_pool_by_iter, gap_hops_flat
            )
        else:
            retrieved_docs = self._round_robin_dedup(main_hops)

        return dspy.Prediction(retrieved_docs=retrieved_docs)