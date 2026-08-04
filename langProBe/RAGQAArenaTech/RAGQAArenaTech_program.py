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

    Query generation includes an intent-clarification step for colloquial/ambiguous
    questions (infer the concrete feature/symptom the user is seeing rather than a
    literal reading of the words), plus a non-negative fallback to the question on
    unusable/refusal queries. Synthesis is a single pass over the accumulated
    context, with a placeholder/parse-error retry. max_hops=3, num_docs=12.
    """

    def __init__(self, retriever, num_docs=12, max_hops=3):
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

    def _is_usable_query(self, query):
        if not query or not query.strip():
            return False
        q = query.strip()
        if len(q) < 3:
            return False
        cjk = sum(1 for c in q if "\u4e00" <= c <= "\u9fff")
        if cjk / max(len(q), 1) > 0.3:
            return False
        low = q.lower()
        if any(low.startswith(p) for p in ("i cannot", "i can't", "i'm unable", "sorry", "无法", "你好")):
            return False
        return True

    def _is_placeholder_response(self, response):
        if not response or not response.strip():
            return True
        r = response.strip().lower()
        if len(r) < 40 and any(
            t in r for t in ("{response}", "[response", "<response", "response text", "...", "your response here")
        ):
            return True
        return False

    def _generate_query(self, hop, context, question):
        try:
            q = self.generate_query[hop](context=context, question=question).query
            if self._is_usable_query(q):
                return q
        except Exception:
            pass
        return question

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self._generate_query(hop, context, question)
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + passages)
        # Single synthesis pass. Retry once on a parse
        # exception or a placeholder non-answer (strictly non-negative).
        pred = None
        for _attempt in range(2):
            try:
                pred = self.respond(context=context, question=question)
                if not self._is_placeholder_response(pred.response):
                    break
            except Exception:
                pred = None
        if pred is None:
            pred = dspy.Prediction(response="")
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence.
        pred.context = context
        return pred
