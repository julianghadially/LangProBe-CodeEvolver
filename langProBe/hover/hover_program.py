import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_RETRIEVED_DOCS = 21


class ClaimEntities(dspy.Signature):
    """List every distinct named entity mentioned in the claim — people, works (films, songs,
    albums, books, video games), organizations, places, events, concepts. Output each entity as
    the SPECIFIC name used in the claim, preserving any year, edition or disambiguator
    (e.g. "2005 NASDAQ-100 Open Women's Doubles", "Lolita (1962 film)", "G.I. Joe: Hall of Fame"),
    not a generalized category. Use Wikipedia-style article names."""
    claim = dspy.InputField(desc="The claim")
    entities: list[str] = dspy.OutputField(desc="Named entities in the claim, as specific Wikipedia-style names")


class QueryExpansion(dspy.Signature):
    """You are retrieving Wikipedia documents for a claim. This is PURE DOCUMENT RETRIEVAL —
    do NOT verify, fact-check, or judge whether the claim is true, and do NOT decide the claim is
    "supported". Your only job: output search queries for documents STILL MISSING.

    What "missing" means (read carefully):
    - An entity is COVERED only if one of the retrieved document TITLES is (or matches) that
      entity. An entity merely MENTIONED inside another article's snippet is NOT covered — it
      still needs its own article. (e.g. if an award snippet lists "S. Truett Cathy" as a recipient
      but no retrieved title is "S. Truett Cathy", then "S. Truett Cathy" is missing.)
    - Even entities named directly in the claim may NOT have been retrieved yet. Compare the
      retrieved TITLES to the claim; any claim entity with no matching title is missing.
    - Mine the retrieved SNIPPETS for proper nouns the claim depends on but that lack their own
      article: the winner named in a tournament article, the recipient named in an award article,
      the cast named in a film article, the town a school serves, the director of a video, the
      founder of a company, the band behind an album, the spouse of a person, the subject of a
      biographical film. Output each such name as a query.
    - ALSO use your own world knowledge to infer entities the claim refers to that are not yet
      in the retrieved titles, even if they are not explicitly mentioned in the snippets.
      For example: if the claim mentions "The Broken Tower" (a film), use your knowledge that
      it is about the poet "Hart Crane"; if the claim mentions a person who co-founded a city,
      use your knowledge of their spouse if the claim depends on them.

    Rules:
    - Each query targets ONE entity and is its exact Wikipedia-style title/name, short (1-5
      words), e.g. "Amanda Wyss", "Lisa Raymond", "S. Truett Cathy", "Chick-fil-A",
      "Eighth Wonder", "Bass (voice type)", "Warren Fu", "Vision of the Future".
    - Do NOT write full sentences, questions, or justification/verification text.
    - Do NOT repeat a query whose name already matches a retrieved document title.
    - Cover DISTINCT missing entities; queries must be non-redundant.
    - Do NOT return an empty list unless every entity named in the claim AND every entity the
      claim depends on already has its own retrieved article (by title). When unsure, output a
      query rather than none.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    retrieved_docs = dspy.InputField(desc="Documents retrieved so far as '<title> | <snippet>', one per line")
    queries: list[str] = dspy.OutputField(desc="Concise entity-name search queries for missing documents")


class ClaimAnalysis(dspy.Signature):
    """Analyze the claim to identify entities, works, or concepts it refers to but does NOT
    explicitly name — entities that a supporting Wikipedia document would need to cover.

    This is PURE DOCUMENT RETRIEVAL — do NOT verify, fact-check, or judge whether the claim
    is true. Just identify what the claim is referring to and output Wikipedia-style search
    queries for those entities.

    Examples of what to infer:
    - "The actor who starred in an Oscar winning film with Amber Tamblyn" → the film is
      "127 Hours", so output "127 Hours"
    - "A genus containing Butein" → the genus is "Dahlia", so output "Dahlia"
    - "The followup novel to the Heir to the Empire trilogy" → "Vision of the Future"
    - "The director of Pacific Rim" → "Guillermo del Toro" (if not already named)
    - "The band behind an album" → the band's name

    Rules:
    - Each query is a short Wikipedia-style title/name (1-5 words).
    - Do NOT repeat entities already explicitly named in the claim.
    - Do NOT verify whether the claim is true. Just identify what it refers to.
    - When unsure, output a query rather than nothing — missing documents cost the entire score.
    """
    claim = dspy.InputField(desc="The claim to analyze")
    queries: list[str] = dspy.OutputField(desc="Wikipedia-style search queries for entities the claim refers to")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25
        self.hop2_queries = 8
        self.hop3_queries = 4
        self.hop4_queries = 3
        self.hop1_keep = 12
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_entities = dspy.Predict(ClaimEntities)
        self.expand_queries = dspy.ChainOfThought(QueryExpansion)
        self.analyze_claim = dspy.ChainOfThought(ClaimAnalysis)

    @staticmethod
    def _title(doc):
        return doc.split(" | ", 1)[0].strip().lower()

    @staticmethod
    def _round_robin(lists):
        seen = set()
        out = []
        max_len = max((len(lst) for lst in lists), default=0)
        for rank in range(max_len):
            for lst in lists:
                if rank < len(lst):
                    doc = lst[rank]
                    t = HoverMultiHop._title(doc)
                    if t not in seen:
                        seen.add(t)
                        out.append(doc)
        return out

    def _retrieve_many(self, queries):
        return [self.retrieve_k(q).passages for q in queries]

    def _context_docs(self, docs, n=45, max_snippet=600):
        seen = set()
        uniq = []
        for d in docs:
            t = self._title(d)
            if t not in seen:
                seen.add(t)
                uniq.append(d)
        lines = []
        for d in uniq[:n]:
            parts = d.split(" | ", 1)
            title = parts[0]
            snippet = parts[1][:max_snippet].replace("\n", " ") if len(parts) > 1 else ""
            lines.append(f"{title} | {snippet}")
        return "\n".join(lines)

    def _build_queries(self, raw, seen_queries, limit):
        out = []
        for q in (raw or []):
            q = (q or "").strip()
            ql = q.lower()
            if q and ql not in seen_queries:
                seen_queries.add(ql)
                out.append(q)
            if len(out) >= limit:
                break
        return out

    def _safe_entities(self, claim):
        try:
            return getattr(self.extract_entities(claim=claim), "entities", None) or []
        except Exception:
            return []

    def _safe_expand(self, claim, context):
        try:
            return getattr(self.expand_queries(claim=claim, retrieved_docs=context), "queries", None) or []
        except Exception:
            return []

    def _safe_analyze(self, claim):
        try:
            return getattr(self.analyze_claim(claim=claim), "queries", None) or []
        except Exception:
            return []

    def _select(self, hop1, other_lists):
        """Dilution-robust selection: always keep hop1's top docs (claim-named golds land here)
        and the top hit of every query (directly-targeted / discovered golds land here), then fill
        the rest with round-robin for breadth. This keeps claim-named golds regardless of how many
        query-lists are used, so extra queries can't dilute them out."""
        seen = set()
        keep = []

        def add(doc):
            t = self._title(doc)
            if t not in seen:
                seen.add(t)
                keep.append(doc)

        for d in hop1[: self.hop1_keep]:
            add(d)
        for lst in other_lists:
            if lst:
                add(lst[0])
        for d in self._round_robin([hop1] + other_lists):
            if len(keep) >= MAX_RETRIEVED_DOCS:
                break
            add(d)
        return keep[:MAX_RETRIEVED_DOCS]

    def forward(self, claim):
        seen_queries = {claim.strip().lower()}
        hop1 = self.retrieve_k(claim).passages

        all_docs = list(hop1)
        other_lists = []

        # Claim-named entities that are NOT already in hop1's top results are queried directly so
        # they land at rank 1. Entities already in hop1's top results are kept by _select, so we
        # skip them to avoid adding redundant query-lists.
        hop1_top_titles = {self._title(d) for d in hop1[: self.hop1_keep]}
        claim_ents = [e for e in self._safe_entities(claim)
                      if self._title(e) not in hop1_top_titles]

        # Hop 2: claim-named entities first, then claim analysis queries (LM knowledge to infer
        # what the claim refers to), then snippet-mined expansion queries.
        context = self._context_docs(all_docs)
        hop2_q = self._build_queries(
            claim_ents + self._safe_analyze(claim) + self._safe_expand(claim, context),
            seen_queries, self.hop2_queries
        )
        if hop2_q:
            hop2_lists = self._retrieve_many(hop2_q)
            other_lists.extend(hop2_lists)
            for lst in hop2_lists:
                all_docs.extend(lst)

        # Hop 3: expansion that mines hop 2's newly retrieved snippets for further entities
        # (e.g. a song article naming its video director, an album article naming its band).
        context = self._context_docs(all_docs)
        hop3_q = self._build_queries(
            self._safe_expand(claim, context), seen_queries, self.hop3_queries
        )
        if hop3_q:
            hop3_lists = self._retrieve_many(hop3_q)
            other_lists.extend(hop3_lists)
            for lst in hop3_lists:
                all_docs.extend(lst)

        # Hop 4: final expansion pass to catch deeply-nested entities
        context = self._context_docs(all_docs)
        hop4_q = self._build_queries(
            self._safe_expand(claim, context), seen_queries, self.hop4_queries
        )
        if hop4_q:
            hop4_lists = self._retrieve_many(hop4_q)
            other_lists.extend(hop4_lists)
            for lst in hop4_lists:
                all_docs.extend(lst)

        final = self._select(hop1, other_lists)
        return dspy.Prediction(retrieved_docs=final)
