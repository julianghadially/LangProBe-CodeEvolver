"""Pipeline wrapper for RAGQAArenaTech, mirroring HoverMultiHopPipeline.

This thin wrapper exists so the CodeEvolver IterationArchitect can modify or
swap out the inner RAG program (default: SimplifiedBaleen) -- or insert/compose
additional modules before it -- without touching the evaluation harness. The
wrapper owns LM configuration, the retrieval database (injected into the program),
and a stable ``forward(question)`` interface; the inner program is pure logic.
"""

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .RAGQAArenaTech_program import SimplifiedBaleen
from .RAGQAArenaTech_retrieval import get_default_retriever

# DeepSeek-V4-Flash hosted on DeepInfra, routed through LiteLLM/DSPy. No
# reasoning_effort is set, so the provider's default ("normal") effort is used.
# The DeepInfra key is read by LiteLLM from the DEEPINFRA_API_KEY env var at call
# time -- never passed into dspy.LM(...), so it stays out of the OTel trace files.
MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"
MODEL = "openai/gpt-5.4-nano"

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

    Retrieval (OpenAI text-embedding-3-small over a precomputed index.pt) lives in
    an injected ``EmbeddingRetriever`` that the pipeline owns and passes into the
    program -- the program itself does no IO. The retriever is a process-wide
    singleton, so the 3.9GB index loads once and is shared.
    """

    def __init__(self, program: dspy.Module | None = None, retriever=None):
        super().__init__()
        self.lm = dspy.LM(MODEL, reasoning_effort="low")
        # The pipeline owns the retrieval database and injects it into the program,
        # so the inner program is pure logic the optimizer can freely swap/evolve.
        self.retriever = retriever if retriever is not None else get_default_retriever()
        self.program = (
            program if program is not None else SimplifiedBaleen(self.retriever)
        )

    def forward(self, question):
        with dspy.context(lm=self.lm):
            return self.program(question=question)
