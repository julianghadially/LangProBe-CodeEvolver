import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hotpot_program import HotpotMultiHopPredict

# Import SerperService directly to avoid circular import in services/__init__.py
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("serper_service", "/workspace/services/serper_service.py")
serper_module = importlib.util.module_from_spec(spec)
sys.modules["serper_service"] = serper_module
spec.loader.exec_module(serper_module)
SerperService = serper_module.SerperService

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"

class HotpotMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Pipeline with hybrid retrieval (Wikipedia + Web Search)."""

    def __init__(self):
        super().__init__()
        # ColBERT retriever for hop 1 (Wikipedia)
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Serper service for hop 2 (Web search)
        self.serper_service = SerperService()

        # Program with injected service
        self.program = HotpotMultiHopPredict(serper_service=self.serper_service)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
