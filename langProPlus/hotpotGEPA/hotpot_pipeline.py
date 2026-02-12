import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHopPredict


class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Pipeline for multi-hop question answering using web search and scraping."""

    def __init__(self):
        super().__init__()
        self.program = HotpotMultiHopPredict()

    def forward(self, question):
        return self.program(question=question)
