import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHopPredict


class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """
    Pipeline for HotPot Multi-Hop QA using Serper + Firecrawl.

    No longer requires ColBERTv2 - retrieval is handled internally
    by HotpotMultiHopPredict using SerperService and FirecrawlService.
    """

    def __init__(self):
        super().__init__()
        # Services (Serper, Firecrawl) are initialized in HotpotMultiHopPredict
        self.program = HotpotMultiHopPredict()

    def forward(self, question):
        # No retrieval model context needed - handled internally
        return self.program(question=question)
