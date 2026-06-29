import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .RAGQAArenaTech_utils import GenerateSearchQuery


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
        self.respond = dspy.ChainOfThought("context, question -> response")
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
        return self.respond(context=context, question=question)
