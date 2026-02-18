import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class HotpotMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Adapted from HoverMultiHop. Hop 3 replaced with answer generation."""

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HotpotMultiHop()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)

