from openinference.instrumentation.dspy import DSPyInstrumentor

DSPyInstrumentor().instrument()

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .counting_rm import CountingRM
from .hotpot_program import HotpotMultiHop, HotpotMultiHopPredict

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"


class HotpotMultiHopProgram:
    """CodeEvolver-compatible wrapper around the Hotpot multi-hop DSPy pipeline.

    Instantiated once per sandbox start. OpenTelemetry DSPy spans, LM configuration,
    and pipeline assembly happen in ``__init__``. Called per-example via ``__call__``.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        colbert_url: str = COLBERT_URL,
    ):
        self.lm = dspy.LM(model)
        dspy.configure(lm=self.lm)

        self.rm = CountingRM(dspy.ColBERTv2(url=colbert_url))
        self.program = HotpotMultiHop()

    def __call__(self, question: str):
        self.rm.reset_count()
        with dspy.context(rm=self.rm):
            result = self.program(question=question)
        result.retrieval_count = self.rm.get_count()
        return result


class HotpotMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Adapted from HoverMultiHop. Hop 3 replaced with answer generation."""

    def __init__(self):
        super().__init__()
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HotpotMultiHop()

    def forward(self, question):
        self.rm.reset_count()
        with dspy.context(rm=self.rm):
            result = self.program(question=question)
        result.retrieval_count = self.rm.get_count()
        return result


class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HotpotMultiHopPredict()

    def forward(self, question):
        self.rm.reset_count()
        with dspy.context(rm=self.rm):
            result = self.program(question=question)
        result.retrieval_count = self.rm.get_count()
        return result
