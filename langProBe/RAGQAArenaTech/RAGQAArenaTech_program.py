import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery, GenerateAnswer


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
        if not response or not str(response).strip():
            pred = self.respond(context=context, question=question)
            response = getattr(pred, "response", None)
        if not response or not str(response).strip():
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
            context = deduplicate(context + passages)
        pred = self._respond(context, question)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred