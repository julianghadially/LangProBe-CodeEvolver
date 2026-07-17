import re
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

# Literal placeholder/stub responses the LM occasionally emits without raising
# AdapterParseError (e.g. "{response}", "...", empty). These are catastrophic
# zero-score outputs the exception-only guard above misses, because the adapter
# successfully parsed *a* response -- it just wasn't a real answer. The cheaper
# `dspy.Predict` fallback (no `reasoning` field) on re-roll skips the stub path.
_PLACEHOLDER_TOKENS = {
    "[answer]",
    "[no content]",
}


def _is_malformed_response(text) -> bool:
    """True if a synthesis `response` is a catastrophic stub/placeholder.

    Only fires on definitively-broken outputs: empty/whitespace, a literal
    template token like ``{response}``, an explicit ``[answer]`` marker, or a
    stub of only dots (``...``). Short lexical answers like ``None``/``No``/
    ``N/A`` that can be valid are intentionally NOT triggered, so real
    (including terse) answers are byte-for-byte unchanged.
    """
    if text is None:
        return True
    t = re.sub(r"\s+", "", str(text)).lower()
    if not t:
        return True
    if t in _PLACEHOLDER_TOKENS:
        return True
    # Bare ``{template_variable}`` form for any name (covers adapter stubs).
    if re.fullmatch(r"\{[a-z_][a-z0-9_]*\}", t):
        return True
    # A stub of only dots (``.``, ``...``, ``......``): the adapter parsed
    # a placeholder, not a real answer.
    if re.fullmatch(r"\.{1,}", t):
        return True
    return False


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

        Two catastrophic triggers route to the cheaper `dspy.Predict` fallback:
          (a) the primary `ChainOfThought` raises a retriable exception (the
              LM-gateway hard-cap), OR
          (b) the primary call returns a *content stub* response (e.g.
              ``{response}``, ``...``, empty) without raising -- the adapter
              parsed a placeholder, not a real answer.
        Both share the same cheaper-shape fallback (drop `reasoning`). Falls
        back to raw `question` only if the cheaper Predict also fails / stubs.
        """
        try:
            pred = self.respond(context=context, question=question)
        except Exception as exc:  # noqa: BLE001 - broad catch is deliberate for recovery
            if not _is_retriable(exc):
                raise
            pred = None
        if pred is not None and not _is_malformed_response(getattr(pred, "response", None)):
            return pred
        # Either primary raised a retriable exc, or returned a content stub:
        # re-roll with the cheaper `dspy.Predict` (no `reasoning` field) which
        # fits the LM-gateway budget and skips the stub-emission path.
        if _RETRY_BACKOFF:
            time.sleep(_RETRY_BACKOFF)
        try:
            pred = self.respond_fallback(context=context, question=question)
        except Exception as exc2:  # noqa: BLE001
            if not _is_retriable(exc2):
                raise
            return dspy.Prediction(response=question)
        if _is_malformed_response(getattr(pred, "response", None)):
            return dspy.Prediction(response=question)
        return pred

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