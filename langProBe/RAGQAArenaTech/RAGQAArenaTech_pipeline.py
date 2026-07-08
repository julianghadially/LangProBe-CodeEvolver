"""Pipeline wrapper for RAGQAArenaTech, mirroring HoverMultiHopPipeline.

This thin wrapper exists so the CodeEvolver IterationArchitect can modify or
swap out the inner RAG program (default: SimplifiedBaleen) -- or insert/compose
additional modules before it -- without touching the evaluation harness. The
wrapper owns LM configuration, the retrieval database (injected into the program),
and a stable ``forward(question)`` interface; the inner program is pure logic.
"""

import os

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .RAGQAArenaTech_program import SimplifiedBaleen
from .RAGQAArenaTech_retrieval import get_default_retriever

# Arm DeepSeek-V4-Flash through GMI Cloud, routed through LiteLLM/DSPy, using LiteLLM's OpenAI-compatible route:
# model="openai/<id>" + api_base=<GMI endpoint>.
# Note reasoning is enabled via the standard OpenAI `reasoning_effort` param (tested with mlflow)
# The GMI key MUST be passed explicitly otherwise it would fall back to OPENAI_API_KEY.
# Note: the key is redacted from traces by DSPy+OpenInference. checked with Opentelemetry.
MODEL = "openai/deepseek-ai/DeepSeek-V4-Flash"
GMI_API_BASE = "https://api.gmi-serving.com/v1"

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
        self.lm = dspy.LM(
            MODEL,
            api_base=GMI_API_BASE,
            api_key=os.environ["GMI_API_KEY"],
            reasoning_effort="high",
            allowed_openai_params=["reasoning_effort"],
        )
        # The pipeline owns the retrieval database and injects it into the program,
        # so the inner program is pure logic the optimizer can freely swap/evolve.
        self.retriever = retriever if retriever is not None else get_default_retriever()
        self.program = (
            program
            if program is not None
            else SimplifiedBaleen(self.retriever, num_docs=8, max_hops=3)
        )

    def forward(self, question):
        with dspy.context(lm=self.lm):
            return self.program(question=question)
