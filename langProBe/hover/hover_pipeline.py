import langProBe.hover.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import

import os

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .counting_rm import CountingRM
from .hover_program import HoverMultiHop

# Disable DSPy cache so every run exercises the real LM and ColBERT calls.
try:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except AttributeError:
    pass  # older DSPy without configure_cache

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"

# Arm DeepSeek-V4-Flash through GMI Cloud, routed through LiteLLM/DSPy, using LiteLLM's OpenAI-compatible route: 
# model="openai/<id>" + api_base=<GMI endpoint>. 
# Note reasoning is enabled via the standard OpenAI `reasoning_effort` param (tested with mlflow)
# The GMI key MUST be passed explicitly otherwise it would fall back to OPENAI_API_KEY.
# Note: the key is redacted from traces by DSPy+OpenInference. checked with Opentelemetry.
MODEL = "openai/deepseek-ai/DeepSeek-V4-Flash"
GMI_API_BASE = "https://api.gmi-serving.com/v1"



class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.lm = dspy.LM(
            MODEL,
            api_base=GMI_API_BASE,
            api_key=os.environ["GMI_API_KEY"],
            reasoning_effort="high",
            allowed_openai_params=["reasoning_effort"],
        )

        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HoverMultiHop()

    def forward(self, claim):
        self.rm.reset_count()
        with dspy.context(lm=self.lm, rm=self.rm):
            result = self.program(claim=claim)
        result.search_count = self.rm.get_count()
        return result
