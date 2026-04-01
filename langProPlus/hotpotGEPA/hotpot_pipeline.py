import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .counting_rm import CountingRM
from .hotpot_program import HotpotMultiHop, HotpotMultiHopPredict

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"


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
