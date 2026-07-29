import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery


class GenerateAnswer(dspy.Signature):
    """Answer the user's question directly and helpfully.

    Use the retrieved context as the primary source of facts; where the context is
    silent or incomplete you may also draw on your own knowledge. Every claim must be
    truthful and specific -- never fabricate or speculate.

    Provenance for specifics: for high-stakes precise figures -- exact round counts,
    cryptographic key lengths, byte/bandwidth thresholds, and similar numbers where an
    invented value would mislead -- state them ONLY when they come from the retrieved
    context or you are confident of them; otherwise give the governing principle or
    qualitative guidance ("choose by your target verification time"), not a fabricated
    number. For ordinary, well-established specifics (product names, common commands,
    widely-known defaults) it is fine to draw on your own knowledge.

    Cover all the question-relevant points as completely as a good expert answer would,
    including the concrete names, commands, and numbers in the context. Stay focused and
    concise. Write in plain, natural prose. Do not include bracketed citations, source
    tags, response templates, or any placeholder tokens -- output only the final answer
    itself.
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

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + passages)
        pred = self.respond(context=context, question=question)
        # Carry the retrieved passages on the prediction so downstream metrics
        # (e.g. faithfulness/groundedness) can see the evidence the answer was
        # generated from. Mirrors hover_program.py's dspy.Prediction(retrieved_docs=...).
        pred.context = context
        return pred
