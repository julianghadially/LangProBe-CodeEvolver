import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_RETRIEVED_DOCS = 21


class ClaimEntities(dspy.Signature):
    """List every distinct named entity mentioned in the claim — people, works (films, songs,
    albums, books), organizations, places, events, concepts. Output each as its most common
    Wikipedia-style article name (proper noun, disambiguated, e.g. "Lolita (1962 film)")."""
    claim = dspy.InputField(desc="The claim")
    entities: list[str] = dspy.OutputField(desc="Named entities in the claim, as Wikipedia-style names")


class QueryExpansion(dspy.Signature):
    """You are retrieving Wikipedia documents for a claim. This is PURE DOCUMENT RETRIEVAL —
    do NOT verify, fact-check, or judge whether the claim is true, and do NOT decide the claim
    is "supported". Your only job: output search queries for documents STILL MISSING.

    What "missing" means (read carefully):
    - An entity is COVERED only if one of the retrieved document TITLES is (or matches) that
      entity. An entity that is merely MENTIONED inside another article's snippet is NOT covered
      — it still needs its own article retrieved. (e.g. if an award snippet lists "S. Truett
      Cathy" as a recipient but no retrieved title is "S. Truett Cathy", then "S. Truett Cathy"
      is missing.)
    - Even entities named directly in the claim may NOT have been retrieved yet. Check the
      retrieved TITLES against the claim; any claim entity with no matching title is missing.
    - Mine the retrieved SNIPPETS for proper nouns the claim depends on but that lack their own
      article: the winner named in a tournament article, the recipient named in an award
      article, the cast named in a film article, the town a school serves, the director of a
      video, the founder of a company, the band behind an album. Output each such name as a query.

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


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20
        self.num_queries = 5
        self.max_per_cluster = 3
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_entities = dspy.Predict(ClaimEntities)
        self.expand_queries = dspy.ChainOfThought(QueryExpansion)

    @staticmethod
    def _title(doc):
        return doc.split(" | ", 1)[0].strip().lower()

    @staticmethod
    def _cluster_key(title):
        t = title.lower()
        t = re.sub(r"\([^)]*\)", "", t)
        t = re.sub(r"^\d{4}[\u2013\-]\s*\d{0,4}\s*", "", t)  # leading "1979-80 " or "2005-"
        t = re.sub(r"^\d{4}\s+", "", t)  # leading "2005 "
        t = re.sub(r"[^a-z0-9]+", " ", t).strip()
        return t

    @staticmethod
    def _round_robin_dedup(lists):
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

    def _diversity_cap(self, docs, max_per_cluster):
        counts = {}
        out = []
        for d in docs:
            key = self._cluster_key(d.split(" | ", 1)[0])
            if counts.get(key, 0) < max_per_cluster:
                counts[key] = counts.get(key, 0) + 1
                out.append(d)
        return out

    def _retrieve_many(self, queries):
        return [self.retrieve_k(q).passages for q in queries]

    def _context_docs(self, docs, n=25, max_snippet=400):
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

    def forward(self, claim):
        seen_queries = {claim.strip().lower()}
        hop1 = self.retrieve_k(claim).passages

        retrieved_titles = set(self._title(d) for d in hop1)
        all_docs = list(hop1)
        retrieval_lists = [hop1]

        # Robust base queries: always retrieve own articles for entities named in the claim
        # that were not already retrieved by hop 1. This cannot be suppressed by the expansion
        # LM's verification bias, and catches claim-named gold docs (e.g. "Amanda Wyss").
        claim_ents = [e for e in self._safe_entities(claim)
                      if self._title(e) not in retrieved_titles]

        # Hop 2: claim-named entities first, then snippet-mined expansion queries.
        context = self._context_docs(all_docs)
        hop2_queries = self._build_queries(
            claim_ents + self._safe_expand(claim, context), seen_queries, self.num_queries
        )
        if hop2_queries:
            hop2_lists = self._retrieve_many(hop2_queries)
            retrieval_lists.extend(hop2_lists)
            for lst in hop2_lists:
                all_docs.extend(lst)

        # Hop 3: expansion that mines hop 2's newly retrieved snippets for further entities
        # (e.g. a song article naming its video director, an album article naming its band).
        context = self._context_docs(all_docs)
        hop3_queries = self._build_queries(
            self._safe_expand(claim, context), seen_queries, self.num_queries
        )
        if hop3_queries:
            hop3_lists = self._retrieve_many(hop3_queries)
            retrieval_lists.extend(hop3_lists)
            for lst in hop3_lists:
                all_docs.extend(lst)

        candidates = self._round_robin_dedup(retrieval_lists)
        candidates = self._diversity_cap(candidates, self.max_per_cluster)
        final = candidates[:MAX_RETRIEVED_DOCS]
        return dspy.Prediction(retrieved_docs=final)
