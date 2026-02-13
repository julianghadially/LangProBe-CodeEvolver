import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHopPredict
from .web_retrieval import WikipediaWebRetrieval


class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.rm = WikipediaWebRetrieval(k=7, max_content_length=10000)
        self.program = HotpotMultiHopPredict()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
