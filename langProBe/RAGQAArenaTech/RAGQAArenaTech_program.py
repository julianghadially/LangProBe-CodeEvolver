import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import (
    GenerateDiverseSearchQueries,
    AssessPassageRelevance,
)


class SimplifiedBaleen(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop RAG program for RAGQAArenaTech.

    Retrieval is injected (see ``RAGQAArenaTech_retrieval.HTTPEmbeddingRetriever``):
    the program holds no corpus/index and performs no IO -- it calls out to a
    separate warm retriever server. This keeps the program pure logic the optimizer
    can freely evolve (hops, query generation, answer synthesis), while the heavy
    3.7GB embedding index lives in a process loaded once, outside the eval.

    Iteration 3: per-hop *diverse multi-query retrieval* (a recall knob) followed by
    an *LLM relevance rerank/keep* gate before synthesis (a precision knob). Each hop
    emits up to ``n_queries`` sub-queries via ``GenerateDiverseSearchQueries``;
    each sub-query retrieves ``k_per_query`` passages which are merged+deduped into a
    capped (``max_context``) context. After all hops, ``_rerank_context`` uses the LM
    to score each passage's relevance and keeps the top ``keep`` for synthesis.
    """

    def __init__(
        self,
        retriever,
        num_docs=5,
        max_hops=2,
        n_queries=3,
        k_per_query=3,
        max_context=12,
        rerank_keep=10,
    ):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.num_docs = num_docs
        self.n_queries = n_queries
        self.k_per_query = k_per_query
        self.max_context = max_context
        self.rerank_keep = rerank_keep
        self.respond = dspy.ChainOfThought("context, question -> response")
        # Per-hop diverse query generation (recall knob).
        self.generate_queries = [
            dspy.ChainOfThought(GenerateDiverseSearchQueries)
            for _ in range(self.max_hops)
        ]
        # LLM relevance rerank before synthesis (precision gate).
        self.rerank = dspy.ChainOfThought(AssessPassageRelevance)

    def search(self, query, k=5):
        return self.retriever.search(query, k=k)

    @staticmethod
    def _parse_queries(raw):
        """Parse a raw ``queries`` LM output into a clean list of query strings.

        Robust to commas, newlines, or numbered/bulleted lists. Returns the unique,
        non-empty stripped queries in first-seen order.
        """
        if raw is None:
            return []
        text = str(raw).strip()
        if not text:
            return []
        # Split on newlines first (the signature asks for newline-separated).
        parts = re.split(r"[\n\r]+", text)
        # If a single line came back with commas, split on commas too.
        if len(parts) == 1 and "," in parts[0]:
            parts = parts[0].split(",")
        queries = []
        seen = set()
        for part in parts:
            q = part.strip()
            # Strip leading list markers / numbering (e.g. "1.", "-", "*").
            q = re.sub(r"^(?:\d+[.)]|[-*+]|[A-Za-z][.)])\s*", "", q).strip()
            # Strip surrounding quotes.
            if (q.startswith('"') and q.endswith('"')) or (
                q.startswith("'") and q.endswith("'")
            ):
                q = q[1:-1].strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        return queries

    def _rerank_context(self, question, passages, keep=8):
        """LLM relevance rerank/keep gate before synthesis.

        Scores each passage with ``AssessPassageRelevance``, then returns the top
        ``keep`` passages (original objects, preserving stable tie order). On any
        failure, falls back to ``passages[:keep]`` -- never lets rerank crash an eval.
        """
        try:
            scored = []
            for idx, passage in enumerate(passages):
                passage_text = getattr(passage, "text", str(passage))
                passage_text = str(passage_text)
                if len(passage_text) > 1500:
                    passage_text = passage_text[:1500]
                out = self.rerank(question=question, passage=passage_text)
                score = self._parse_relevance_score(out.score)
                # Stable sort key: (score desc, original idx asc) preserves input
                # order for ties so rerank is deterministic.
                scored.append((idx, passage, score))
            scored.sort(key=lambda x: (-x[2], x[0]))
            return [p for _, p, _ in scored[:keep]]
        except Exception:
            return list(passages)[:keep]

    @staticmethod
    def _parse_relevance_score(raw):
        """Robustly parse the rerank score into an int in 0..2 (default 1)."""
        if raw is None:
            return 1
        for tok in re.findall(r"-?\d+", str(raw)):
            try:
                v = int(tok)
            except ValueError:
                continue
            if 0 <= v <= 2:
                return v
        return 1

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            gen = self.generate_queries[hop](context=context, question=question)
            queries = self._parse_queries(gen.queries)[: self.n_queries]
            if not queries:
                # Never skip retrieval -- fall back to the raw question.
                queries = [question]
            for q in queries:
                passages = self.search(q, k=self.k_per_query)
                context = deduplicate(context + passages)
                if len(context) >= self.max_context:
                    context = context[: self.max_context]
                    break
            context = context[: self.max_context]
        # Precision gate: keep only the most LM-relevant passages for synthesis.
        context = self._rerank_context(question, context, keep=self.rerank_keep)
        return self.respond(context=context, question=question)