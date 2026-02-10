import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHop, HotpotMultiHopPredict

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"

class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HotpotMultiHopPredict()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
