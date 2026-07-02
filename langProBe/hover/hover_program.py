import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_DOCS = 21

MAX_GAP_TITLES = 3
GAP_PASSAGES_PER_TITLE = 3
PASSAGE_SNIPPET_CHARS = 600
SNIPPET_DOCS_PER_HOP = 5


class IdentifyMissing(dspy.Signature):
    """Identify Wikipedia articles that are needed to verify the claim but are
    NOT yet present in the retrieved document set.

    Inputs:
      - claim: the factual claim we are fact-checking.
      - retrieved_titles: the canonical Wikipedia article titles already retrieved
        (NONE of these should be re-suggested).
      - passage_snippets: LITERAL truncated text excerpts (title + opening sentences)
        from a sample of the already-retrieved Wikipedia articles. These are the
        ground-truth surface evidence — match entity MENTIONS by text span, not gist.

    Reasoning steps:
      1. Enumerate every distinct entity named in the claim (people, organizations,
         works/titles, places, dates-as-eras). For each, decide whether its dedicated
         standalone Wikipedia article is already in `retrieved_titles`. Match by
         EXACT canonical title only — a disambiguated near-name such as
         "Dave Evans (singer)" covers ONLY that sense; do not assume "Dave Evans"
         alone is the same article.
      2. Critical: scan `passage_snippets` for any named entity (person, org, work,
         place) that is MENTIONED inside a passage but whose own standalone article
         is NOT in `retrieved_titles`. Voice your entity extraction aloud, citing
         the snippet fragments where the mention occurs. The most common
         supporting-fact miss is a "bridge" entity that appears only by surname or
         partial name inside another retrieved doc (e.g. a band article that opens
         "frontman Billy Corgan", a filmography listing that names an obscure
         source-work). Surface these mentions even when the claim only refers to
         them indirectly.
      3. Also extract works/roles/items NAMED WITHIN passage snippets —
         filmography/discography entries, cast lists, "remake of"/spin-off/source
         works, season rosters, "directed by", "fronted by", "founded by" — even
         when the claim references them indirectly.
      4. SPECIFIC OVER GENERIC: the claim's "bridge" entity is almost always the
         SPECIFIC named work/show/episode/role/organization — e.g. the named TV
         series "Ishqbaaz", the named album "Marzemino", the named film "Sing
         Street". Generic infrastructure around it — the network/channel/studio/
         publisher/platform that aired/published/released it (Star Plus, BBC,
         Columbia Records, 21st Century Fox, etc.) is rarely the missing bridge;
         prefer the specific named thing even when the snippet only mentions the
         channel. Do NOT add a network/platform/studio title unless it is itself
         the claimed bridge.
      5. AIM FOR COMPLETENESS but PRECISION FIRST: only suggest titles you are
         reasonably confident exist as standalone Wikipedia articles. Listing
         multiple plausible candidates (alternate name-forms, different-sense
         disambiguations) is BETTER than one exact guess — but never INVENT a
         title that is not a real Wikipedia article, and never echo one already in
         `retrieved_titles`.

    Output:
      - missing_titles: a comma-separated list of up to 3 canonical Wikipedia
        article titles whose dedicated article should be retrieved next. Order by
        how likely the article is the claim's missing bridge. If nothing is
        missing, output an empty string. Output ONLY titles already-mentioned-in
        evidence; do not invent.
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Canonical Wikipedia titles already retrieved, one per line. NEVER re-suggest these.")
    passage_snippets: str = dspy.InputField(desc="Literal truncated text excerpts from a sample of retrieved articles. Scan these for entity MENTIONS by text span.")

    missing_titles: str = dspy.OutputField(desc="Comma-separated canonical Wikipedia titles to retrieve next, up to 3. Empty if none.")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.identify_missing = dspy.ChainOfThought(IdentifyMissing)

    @staticmethod
    def _doc_title(passage):
        return passage.split(" | ")[0]

    def _round_robin_dedup(self, hop_docs):
        """Round-robin across hops (each ColBERT-ranked), dedup by title, cap at MAX_DOCS.

        Balanced representation per hop avoids one hop monopolising the budget, so refined
        follow-up hops get their fair share of the 21 slots allotted to search candidates."""
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

        main_hop_docs = [hop1_docs, hop2_docs, hop3_docs]

        # GAP HOP — surface missing bridge articles by scanning RAW passage snippets
        retrieved_titles = []
        seen_titles = set()
        for hop_docs in main_hop_docs:
            for passage in hop_docs:
                title = self._doc_title(passage)
                if title not in seen_titles:
                    seen_titles.add(title)
                    retrieved_titles.append(title)
        titles_str = "\n".join(retrieved_titles) if retrieved_titles else ""

        try:
            gap_resp = self.identify_missing(
                claim=claim,
                retrieved_titles=titles_str,
                passage_snippets=self._build_snippets(main_hop_docs),
            )
            missing_titles_raw = gap_resp.missing_titles or ""
        except Exception:
            missing_titles_raw = ""

        gap_hop = []
        seen_gap_titles = set()
        for raw in [t.strip() for t in missing_titles_raw.replace("\n", ",").split(",")]:
            if not raw or raw in seen_titles:
                continue
            seen_gap_titles.add(raw)
            try:
                gap_docs = self.retrieve_k(raw).passages
            except Exception:
                gap_docs = []
            gap_hop.extend(gap_docs[:GAP_PASSAGES_PER_TITLE])
            if len(seen_gap_titles) >= MAX_GAP_TITLES:
                break

        merged_hops = main_hop_docs + ([gap_hop] if gap_hop else [])
        return dspy.Prediction(retrieved_docs=self._round_robin_dedup(merged_hops))