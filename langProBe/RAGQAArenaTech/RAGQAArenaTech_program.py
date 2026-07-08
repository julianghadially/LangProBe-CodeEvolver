import re

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery, GenerateIntentSearchQuery, GenerateAnswer

# Some completions return the adapter's *literal* template placeholder (e.g.
# "{response text}", "{response}", "{answer}") instead of filling in the field.
# These strip to a non-empty string so the `not response` guard misses them, and
# the metric dumps the placeholder as the system answer -- a guaranteed loss. A
# short, single brace-wrapped token like these is never a real answer.
_PLACEHOLDER_RE = re.compile(r"^\{[^{}\n]{0,40}\}\s*$")


def _is_blank_or_placeholder(text):
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    return bool(_PLACEHOLDER_RE.match(s))


class SimplifiedBaleen(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop RAG program for RAGQAArenaTech.

    Retrieval is injected (see ``RAGQAArenaTech_retrieval.HTTPEmbeddingRetriever``):
    the program holds no corpus/index and performs no IO -- it calls out to a
    separate warm retriever server. This keeps the program pure logic the optimizer
    can freely evolve (hops, query generation, answer synthesis), while the heavy
    3.7GB embedding index lives in a process loaded once, outside the eval.
    """

    def __init__(self, retriever, num_docs=5, max_hops=2, intent_docs=4):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.num_docs = num_docs
        # On the first hop we ALSO run an intent-reformulated query to surface
        # passages for colloquial/figurative questions the broad query may miss
        # (e.g. "search for backdoors" -> how to *detect* them). Small k keeps
        # dilution bounded; later hops are untouched so coverage-complete rows
        # are unaffected. Mirrors the validated breadth mechanism (extra
        # passages dedup away on already-covered questions).
        self.intent_docs = intent_docs
        self.respond = dspy.ChainOfThought(GenerateAnswer)
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(self.max_hops)
        ]
        self.intent_query = dspy.ChainOfThought(GenerateIntentSearchQuery)

    def search(self, query, k=5):
        return self.retriever.search(query, k=k)

    def _generate_query(self, hop, context, question):
        """Run one query-generation hop, robust to adapter parse failures.

        Reasoning models occasionally emit an unparseable completion (e.g. raw
        shell syntax) that DSPy's JSONAdapter cannot deserialize into the `query`
        field, raising AdapterParseError. Recover by falling back to the original
        question -- a usable search query -- rather than failing the whole row.
        """
        try:
            q = self.generate_query[hop](context=context, question=question).query
        except Exception:
            q = None
        if not q or not str(q).strip():
            q = question
        return str(q).strip()

    def _generate_intent_query(self, context, question):
        """Run the intent-reformulation query hop, robust to adapter parse failures.

        Mirrors ``_generate_query``'s guard: on any failure or empty output,
        return None so the caller simply skips the extra search rather than
        crashing the row.
        """
        try:
            q = self.intent_query(context=context, question=question).query
        except Exception:
            q = None
        if not q or not str(q).strip():
            return None
        return str(q).strip()

    def _respond(self, context, question):
        """Synthesize the final answer, robust to empty/None `response` fields.

        Some completions return an empty `response` (content routed entirely to
        the reasoning model's native channel), which downstream scoring dumps as a
        raw Prediction repr -- a guaranteed loss. Retry once on a fresh call; if
        still empty, synthesize a minimal grounded one-sentence answer from the
        strongest retrieved passage rather than emitting nothing.
        """
        pred = self.respond(context=context, question=question)
        response = getattr(pred, "response", None)
        if _is_blank_or_placeholder(response):
            pred = self.respond(context=context, question=question)
            response = getattr(pred, "response", None)
        if _is_blank_or_placeholder(response):
            # Last-resort fallback: a concise, grounded sentence from the context.
            if context:
                response = (
                    "Based on the retrieved context, "
                    + str(context[0])[:500].strip()
                )
            else:
                response = "I could not find an answer in the available context."
            pred.response = response
        return pred

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self._generate_query(hop, context, question)
            passages = self.search(query, k=self.num_docs)
            # Hop 0 only: also pull an intent-reformulated query to surface
            # passages for colloquial/figurative questions. Small k bounds
            # dilution; dedup keeps already-covered questions neutral.
            if hop == 0 and self.intent_docs > 0:
                intent_query = self._generate_intent_query(context, question)
                if intent_query and intent_query != query:
                    passages = passages + self.search(
                        intent_query, k=self.intent_docs
                    )
            context = deduplicate(context + passages)
        pred = self._respond(context, question)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred