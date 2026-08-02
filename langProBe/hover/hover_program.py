import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_RETRIEVED_DOCS = 21


class QueryExpansion(dspy.Signature):
    """You are retrieving Wikipedia documents to support a factual claim. You are given the
    claim and the documents retrieved so far (title + short snippet, one per line).

    Output concise SEARCH QUERIES that will retrieve the documents STILL MISSING.

    How to decide what is missing:
    1. Enumerate every distinct entity the claim depends on — people, works, organizations,
       places, events, concepts. Some are named in the claim; many are only IMPLIED and must be
       discovered (e.g. "the winner of X", "the founder of Y", "the school serving place Z").
    2. READ the retrieved snippets: they frequently NAME the missing entities. An award article
       lists its recipients; a tournament article names the winner; a school article names the
       town it serves; a film article names its cast. Extract those names.
    3. Cross-reference the claim with those names. If the claim says "the man who founded a fast
       food chain in Georgia was given the President's Volunteer Service Award" and a retrieved
       snippet lists "S. Truett Cathy" as an awardee, then "S. Truett Cathy" and "Chick-fil-A"
       are missing entities to retrieve.

    Rules:
    - Each query targets ONE entity and is the entity's exact Wikipedia-style title/name, short
      (1-5 words), e.g. "Amanda Wyss", "Lisa Raymond", "S. Truett Cathy", "Chick-fil-A",
      "Vision of the Future", "Bass (voice type)", "Garden City South, New York".
    - Do NOT write full sentences, questions, or verification/justification text.
    - Do NOT repeat a query whose name already matches a retrieved document title.
    - Cover DISTINCT missing entities; queries must be non-redundant with one another.
    """
    claim = dspy.InputField(desc="The claim to find supporting documents for")
    retrieved_docs = dspy.InputField(desc="Documents retrieved so far as '<title> | <snippet>', one per line")
    queries: list[str] = dspy.OutputField(desc="Up to 4 concise entity-name search queries, one per missing entity")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20
        self.num_queries = 4
        self.max_per_cluster = 3
        self.retrieve_k = dspy.Retrieve(k=self.k)
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

    def _context_docs(self, docs, n=25, max_snippet=350):
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

    def forward(self, claim):
        seen_queries = {claim.strip().lower()}
        hop1 = self.retrieve_k(claim).passages

        all_docs = list(hop1)
        retrieval_lists = [hop1]

        for _ in range(2):  # two entity-discovery expansion hops
            context = self._context_docs(all_docs)
            raw = self.expand_queries(claim=claim, retrieved_docs=context).queries
            queries = self._build_queries(raw, seen_queries, self.num_queries)
            if not queries:
                break
            hop_lists = self._retrieve_many(queries)
            retrieval_lists.extend(hop_lists)
            for lst in hop_lists:
                all_docs.extend(lst)

        candidates = self._round_robin_dedup(retrieval_lists)
        candidates = self._diversity_cap(candidates, self.max_per_cluster)
        final = candidates[:MAX_RETRIEVED_DOCS]
        return dspy.Prediction(retrieved_docs=final)
