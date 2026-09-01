"""Pipeline wrapper for RAGQAArenaTech, mirroring HoverMultiHopPipeline.

This thin wrapper exists so the CodeEvolver IterationArchitect can modify or
swap out the inner RAG program (default: SimplifiedBaleen) -- or insert/compose
additional modules before it -- without touching the evaluation harness. The
wrapper owns LM configuration, the retrieval database (injected into the program),
and a stable ``forward(question)`` interface; the inner program is pure logic.
"""

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from langProBe.lm_provider import build_task_lm
from .RAGQAArenaTech_program import SimplifiedBaleen
from .RAGQAArenaTech_retrieval import get_default_retriever

# DSPy caches LM completions in memory AND on disk (~/.dspy_cache) by default.
# CodeEvolver runs this program directly via its mounted evaluator, bypassing the
# langprobe/simple_eval harnesses that disable caching -- so without this, a rerun
# replays cached completions and produces an instantaneous, non-representative
# eval. Disable both so every run exercises the real LM and embedding retrieval.
try:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except AttributeError:
    pass  # older DSPy without configure_cache

class RAGQAArenaTechPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """RAG question-answering pipeline over the LoTTE 'technology' corpus.

    EVALUATION
    - Assessed with SemanticF1: the generated ``response`` is scored for
      recall/precision of key ideas against the gold ``response``.

    Retrieval (OpenAI text-embedding-3-small over a precomputed index.pt) is served
    by a separate warm retriever server; the pipeline owns an injected
    ``HTTPEmbeddingRetriever`` client and passes it into the program -- the program
    itself does no IO. The client is a process-wide singleton over a pooled HTTP
    session. See ``RAGQAArenaTech_retrieval`` / the ``ragqa-retriever-server`` repo.
    """

    def __init__(self, program: dspy.Module | None = None, retriever=None):
        super().__init__()
        # DeepSeek-V4-Flash (reasoning_effort="high") on GMI Cloud, with a
        # per-call fallback to the same model on DeepInfra when GMI answers a
        # 4xx. Provider wiring lives in langProBe/lm_provider.py.
        self.lm = build_task_lm()
        # The pipeline owns the retrieval database and injects it into the program,
        # so the inner program is pure logic the optimizer can freely swap/evolve.
        self.retriever = retriever if retriever is not None else get_default_retriever()
        self.program = (
            program if program is not None else SimplifiedBaleen(self.retriever)
        )

    def forward(self, question):
        with dspy.context(lm=self.lm):
            return self.program(question=question)
