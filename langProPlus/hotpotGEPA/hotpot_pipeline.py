import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService
from .hotpot_program import HotpotMultiHopPredict


class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Web search-based pipeline for multi-hop question answering."""

    def __init__(self):
        """Initialize the web search pipeline.

        The pipeline now instantiates web services instead of ColBERT retriever.
        Services are created here and passed to the program for better lifecycle management.
        """
        super().__init__()

        # Initialize web services
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()

        # Initialize program with services
        self.program = HotpotMultiHopPredict(
            serper_service=self.serper,
            firecrawl_service=self.firecrawl
        )

    def forward(self, question):
        """Execute the pipeline.

        With web search architecture, no context manager is needed since services
        are injected via constructor rather than global context.

        Args:
            question: The multi-hop question to answer

        Returns:
            dspy.Prediction with answer field
        """
        return self.program(question=question)
