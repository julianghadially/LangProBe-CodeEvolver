import time

import dspy
from dspy.utils.exceptions import AdapterParseError

from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery


# Substrings that, when found in an exception's repr/message, indicate a
# transport-level timeout / gateway hard-cap rather than a parse bug. Retrying
# the *same* over-budget ChainOfThought call does not help these (empirically
# exhausted in iters 9-10); the productive recovery is a *cheaper* call shape.
_RETRYABLE_SUBSTRINGS = (
    "timeout",
    "timed out",
    "服务端处理超时",
    "connection reset",
    "connection aborted",
    "connection reset by peer",
    "read timeout",
)

# Backoff (seconds) before the cheaper-fallback retry path.
_RETRY_BACKOFF = 2.0


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, AdapterParseError):
        return True
    msg = (str(exc) + " " + type(exc).__name__).lower()
    return any(s.lower() in msg for s in _RETRYABLE_SUBSTRINGS)


class SimplifiedBaleen(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop RAG program for RAGQAArenaTech.

    Retrieval is injected (see ``RAGQAArenaTech_retrieval.HTTPEmbeddingRetriever``):
    the program holds no corpus/index and performs no IO -- it calls out to a
    separate warm retriever server. This keeps the program pure logic the optimizer
    can freely evolve (hops, query generation, answer synthesis), while the heavy
    3.7GB embedding index lives in a process loaded once, outside the eval.

    Catastrophe recovery: the GMI synthesis gateway occasionally hard-caps a
    ``ChainOfThought`` call ("HTTP服务端处理超时") -- the LM emits the long
    ``reasoning`` field but runs out of gateway budget before the actionable
    output (``response`` / ``query``), so DSPy raises ``AdapterParseError`` with
    only partial fields parsed. Retrying the *same* over-budget call is
    empirically useless (iters 9-10: sign-flipped valset noise). Instead, on a
    retriable failure we fall back to a *cheaper* ``dspy.Predict`` (no
    ``reasoning`` field) that asks the LM directly for the short actionable
    output, which fits inside the gateway budget. Passing rows are byte-for-byte
    unchanged -- the fallback only triggers on an exception.
    """

    def __init__(self, retriever, num_docs=5, max_hops=2):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.num_docs = num_docs

        # Primary (full CoT) synthesis + query generators.
        self.respond = dspy.ChainOfThought("context, question -> response")
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(self.max_hops)
        ]

        # Cheaper fallback predictors -- plain Predict skips the `reasoning`
        # field, so the LM produces a much smaller completion that fits inside
        # the gateway's time budget when the ChainOfThought call hard-caps.
        self.respond_fallback = dspy.Predict("context, question -> response")
        self.generate_query_fallback = [
            dspy.Predict(GenerateSearchQuery) for _ in range(self.max_hops)
        ]

    def search(self, query, k=5):
        return self.retriever.search(query, k=k)

    def _safe_query(self, hop, context, question):
        """Generate a search query with a cheaper-shape fallback on failure.

        Falls back raw `question` only if *both* the primary ChainOfThought and
        the cheaper Predict raise a retriable exception.
        """
        try:
            return self.generate_query[hop](context=context, question=question).query
        except Exception as exc:  # noqa: BLE001 - broad catch is deliberate for recovery
            if not _is_retriable(exc):
                raise
            try:
                if _RETRY_BACKOFF:
                    time.sleep(_RETRY_BACKOFF)
                return self.generate_query_fallback[hop](
                    context=context, question=question
                ).query
            except Exception as exc2:  # noqa: BLE001
                if not _is_retriable(exc2):
                    raise
                return question

    def _safe_respond(self, context, question):
        """Synthesize an answer with a cheaper-shape fallback on failure.

        Falls back to raw `question` only if *both* the primary ChainOfThought
        and the cheaper Predict raise a retriable exception.
        """
        try:
            return self.respond(context=context, question=question)
        except Exception as exc:  # noqa: BLE001 - broad catch is deliberate for recovery
            if not _is_retriable(exc):
                raise
            try:
                if _RETRY_BACKOFF:
                    time.sleep(_RETRY_BACKOFF)
                return self.respond_fallback(context=context, question=question)
            except Exception as exc2:  # noqa: BLE001
                if not _is_retriable(exc2):
                    raise
                return dspy.Prediction(response=question)

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self._safe_query(hop, context=context, question=question)
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + passages)
        pred = self._safe_respond(context=context, question=question)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred