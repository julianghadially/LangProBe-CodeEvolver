import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_DOCS = 21

MAIN_RESERVE = 15
MAX_GAP_TITLES = 3
GAP_PASSAGES_PER_TITLE = 3
PASSAGE_SNIPPET_CHARS = 600
SNIPPET_DOCS_PER_HOP = 5
MAX_GAP_ITERATIONS = 2
RERANK_SNIPPET_CHARS = 600


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


class RerankPassages(dspy.Signature):
    """You are the FINAL SELECTION step of a Wikipedia document-retrieval pipeline
    that supports a factual claim. A candidate pool of Wikipedia articles (each
    with its title and opening-text excerpt) has been retrieved across several
    hops. From this pool you must KEEP the at-most-21 articles most useful as
    SUPPORTING FACTS for the claim, ordered most-relevant first.

    The retrieval metric rewards keeping the articles whose standalone Wikipedia
    page contains a fact used to support (a piece of) the claim. It does NOT reward
    generic background, container, or merely-adjacent articles. Quantity beyond the
    true supporting set is not itself helpful — but a borderline relevant article
    IS worth keeping over a clearly-irrelevant one, because the final budget is 21
    and the supporting set is often composed of SEVERAL less-salient participants.

    Reasoning:
      1. Decompose the claim into the distinct entities / propositions that each
         need a supporting article (the people, works, orgs, places, events, and
         relations the claim asserts).
      2. For each numbered candidate, judge from its TITLE and excerpt whether it
         is the SUBJECT of one of the claim's supporting facts. An article is
         relevant when the claim is ABOUT it, or about a fact it contains.
      3. Prefer the SPECIFIC named entity (a named person, work, place, event)
         over a mere container/channel/parent topic — BUT keep the broad overview
         article when the claim is itself about a broad topic ("X religion",
         "X education").
      4. Keep every candidate that IS the standalone article of a distinct claim
         entity; do NOT drop a borderline candidate merely because a higher-ranked
         candidate also touches the claim. Supporting-fact sets are frequently the
         LESS salient participants: the runner-up, the supporting cast, the
         secondary entity listed last. When in doubt, KEEP it.
      5. Exclude pure background / adjacent articles whose excerpt only
         contextually mentions the topic without being the subject of a claim
         fact, and exclude exact duplicate senses.

    Inputs:
      - claim: the claim being supported.
      - candidate_passages: a numbered list, one per line, each formatted
        "N. Title :: excerpt". Refer to a candidate by its number N (starting at 1).

    Output:
      - ranked_ids: a comma-separated list of the at-most-21 candidate NUMBERS you
        decide to keep, MOST relevant FIRST. Use each number at most once and only
        numbers that appear in the input list. Return as close to 21 as the
        relevance genuinely supports (fewer is acceptable only when the pool is
        smaller or most remaining candidates are clearly irrelevant), but prefer
        filling remaining budget with borderline-relevant candidates over leaving
        it empty.
    """

    claim: str = dspy.InputField()
    candidate_passages: str = dspy.InputField(desc="Numbered candidate articles, one per line, formatted 'N. Title :: excerpt'. Refer to a candidate by its number N (1-indexed).")
    ranked_ids: str = dspy.OutputField(desc="Comma-separated candidate NUMBERS to keep, most relevant first, at most 21, each number at most once, only numbers present in candidate_passages.")


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
        # LM listwise reranker for final passage selection from the pooled
        # candidates. Replaces the rank-position heuristic merge as the
        # selection step; the heuristic merge is retained as an exception
        # fallback so a parse failure degrades to the prior best backbone.
        self.rerank = dspy.ChainOfThought(RerankPassages)

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
        wastes budget.

        Retained as the exception fallback for the LM reranker (which supersedes it
        as the primary selection step)."""
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

    def _rerank_pool(self, claim, main_hops, gap_hops_flat):
        """Final selection via an LM listwise rerank of the deduped union of all
        retrieved candidates (main hops + gap passages), capped at MAX_DOCS.

        Returns ``None`` on total reranker failure so the caller can degrade to the
        proven heuristic merge. When the pool is already <= MAX_DOCS it is returned
        unchanged. Otherwise the LM is given a numbered list ("N. Title :: excerpt")
        and returns ranked candidate numbers; any unfilled slots up to MAX_DOCS are
        padded from the remaining candidates in pool order (main-hop rank first) so
        the cap is always filled when the pool allows it."""
        seen = set()
        candidates = []
        # Main-hop passages first (ColBERT rank order within each hop, hops in
        # hop order), then gap passages in retrieval order — so pool ordering is a
        # reasonable default even before the LM reranks.
        for hop in main_hops:
            for passage in hop:
                title = self._doc_title(passage)
                if title not in seen:
                    seen.add(title)
                    candidates.append(passage)
        for passage in gap_hops_flat:
            title = self._doc_title(passage)
            if title not in seen:
                seen.add(title)
                candidates.append(passage)

        if not candidates:
            return []
        if len(candidates) <= MAX_DOCS:
            return candidates[:MAX_DOCS]

        # Build the numbered candidate list the LM will rank.
        lines = []
        for i, passage in enumerate(candidates, 1):
            if " | " in passage:
                title, body = passage.split(" | ", 1)
            else:
                title, body = passage, ""
            body = body.strip()
            if len(body) > RERANK_SNIPPET_CHARS:
                body = body[:RERANK_SNIPPET_CHARS] + "…"
            lines.append(f"{i}. {title.strip()} :: {body}")
        candidate_str = "\n".join(lines)

        try:
            resp = self.rerank(claim=claim, candidate_passages=candidate_str)
            raw = (resp.ranked_ids or "").strip()
        except Exception:
            return None
        if not raw:
            return None

        # Parse an ordered list of candidate numbers; tolerate stray characters,
        # "N."/"N)" decorations, ranges ("3-7"), and "and"/"/" separators.
        chosen = []
        chosen_titles = set()

        def take(n):
            if not (1 <= n <= len(candidates)):
                return False
            passage = candidates[n - 1]
            title = self._doc_title(passage)
            if title in chosen_titles:
                return False
            chosen_titles.add(title)
            chosen.append(passage)
            return True

        for tok in raw.replace("\n", ",").replace("/", ",").replace(";", ",").split(","):
            tok = tok.strip().lower().lstrip("and ").strip()
            # "3-7" style range
            if "-" in tok:
                a, _, b = tok.partition("-")
                try:
                    lo, hi = int(a), int(b)
                except ValueError:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                for n in range(lo, hi + 1):
                    if take(n) and len(chosen) >= MAX_DOCS:
                        break
                continue
            digits = ""
            for ch in tok:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                take(int(digits))
            if len(chosen) >= MAX_DOCS:
                break

        if not chosen:
            # LM emitted nothing parseable — treat as failure.
            return None

        # Pad any unfilled slots up to MAX_DOCS from remaining candidates in pool
        # order, so the final budget is filled whenever the pool allows it.
        if len(chosen) < MAX_DOCS:
            for passage in candidates:
                title = self._doc_title(passage)
                if title not in chosen_titles:
                    chosen_titles.add(title)
                    chosen.append(passage)
                    if len(chosen) >= MAX_DOCS:
                        break
        return chosen[:MAX_DOCS]

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

            try:
                gap_resp = self.identify_missing(
                    claim=claim,
                    retrieved_titles=titles_str,
                    previously_suggested_titles=prev_str,
                    passage_snippets=snippets,
                )
                missing_titles_raw = (gap_resp.missing_titles or "").strip()
            except Exception:
                missing_titles_raw = ""

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

        # CONTROL: heuristic-merge-only final selection (reranker disabled) to
        # isolate the LM reranker's marginal value on an identical seed.
        if gap_pool_by_iter:
            retrieved_docs = self._main_first_merge(
                main_hops, gap_pool_by_iter, gap_hops_flat
            )
        else:
            retrieved_docs = self._round_robin_dedup(main_hops)

        return dspy.Prediction(retrieved_docs=retrieved_docs)