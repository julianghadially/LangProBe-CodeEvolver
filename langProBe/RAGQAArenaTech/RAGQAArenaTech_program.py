import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery


class GenerateAnswer(dspy.Signature):
    """Answer the user's question directly and helpfully.

    Treat the retrieved context as the primary, authoritative source of facts. Cover
    every question-relevant point as completely as a good expert answer would, including
    the concrete names, commands, and numbers that appear in the context. Never fabricate
    or speculate.

    Provenance for specifics:
      - State exact commands, flags, and shell syntax; filesystem paths; column / field
        definitions; and language or framework convention rules ONLY when they come from
        the retrieved context or you are genuinely confident of them. A specific that is
        wrong is worse than an honest general principle, and reviewers penalize untruthful
        content first. If the context is silent on a specific, give the governing
        principle or qualitative guidance ("the exact command depends on your version /
        app", "either interpretation is possible") rather than inventing a precise value.
      - For ordinary, well-established specifics you are genuinely confident about
        (product names, common high-level concepts), you may draw on your own knowledge.

    Completeness and nuance -- this matters and is rewarded:
      - Surface the relevant caveats, exceptions, and mode / version / app-dependent
        nuances that the retrieved context provides (e.g. "in CBC mode the IV must be
        unpredictable", "the storage path depends on the reader app"). A terse answer
        that drops a relevant qualification is less helpful than one that states it, even
        briefly. Do not omit a caveat present in the context just to be concise.
      - When a term in the question is ambiguous and the context does not pin down one
        meaning, briefly note the interpretations rather than committing to a single
        (possibly wrong) one.

    Write in plain, natural prose. Stay focused and concise. Do not include bracketed
    citations, source tags, response templates, or any placeholder tokens -- output only
    the final answer itself.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    response = dspy.OutputField(desc="a direct, truthful, well-grounded answer")


class SimplifiedBaleen(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop RAG program for RAGQAArenaTech.

    Retrieval is injected (see ``RAGQAArenaTech_retrieval.HTTPEmbeddingRetriever``):
    the program holds no corpus/index and performs no IO -- it calls out to a
    separate warm retriever server. This keeps the program pure logic the optimizer
    can freely evolve (hops, query generation, answer synthesis), while the heavy
    3.7GB embedding index lives in a process loaded once, outside the eval.
    """

    def __init__(self, retriever, num_docs=5, max_hops=2):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought(GenerateAnswer)
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(self.max_hops)
        ]

    def search(self, query, k=5):
        return self.retriever.search(query, k=k)

    def _respond_robust(self, context, question):
        """Answer-generation with a graceful fallback for intermittent LM hiccups.

        DeepSeek-V4-Flash occasionally returns an empty completion that DSPy's JSONAdapter
        surfaces as ``AdapterParseError`` (or a prediction whose ``response`` is empty /
        None). Such a row is guaranteed to lose the pairwise comparison, so we retry a
        couple of times for transient empty-output glitches; if that still fails, we fall
        back to a grounded summary assembled from the top retrieved passages rather than
        emit a broken/empty answer. The retrieved context is always carried on the
        prediction so faithfulness metrics keep working.
        """
        last_exc = None
        for _ in range(3):
            try:
                pred = self.respond(context=context, question=question)
                if getattr(pred, "response", None):
                    return pred
            except Exception as exc:  # AdapterParseError, JSONDecodeError, etc.
                last_exc = exc
        # All retries produced an unusable answer -- assemble a grounded fallback from
        # the retrieved passages so the row still gets a substantive, honest response.
        snippets = [str(p).strip() for p in (context or []) if str(p).strip()]
        if snippets:
            fallback = (
                "Here is what the relevant sources say:\n\n"
                + "\n\n".join(snippets[:3])
                + "\n\n(Summary assembled from the retrieved passages.)"
            )
        else:
            fallback = (
                f"Regarding \"{question}\": the retrieved references did not surface a "
                "specific answer to this question."
            )
        pred = dspy.Prediction(response=fallback)
        if last_exc is not None:
            pred.respond_error = type(last_exc).__name__
        return pred

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + passages)
        pred = self._respond_robust(context=context, question=question)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred
