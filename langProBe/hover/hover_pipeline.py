import langProBe.hover.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .counting_rm import CountingRM
from .hover_program import HoverMultiHop

# DSPy caches LM completions in memory AND on disk (~/.dspy_cache) by default.
# CodeEvolver runs this program directly via its mounted evaluator, bypassing the
# langprobe/simple_eval harnesses that disable caching -- so without this, a rerun
# replays cached completions and produces an instantaneous, non-representative
# eval. Disable both so every run exercises the real LM and ColBERT calls.
try:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except AttributeError:
    pass  # older DSPy without configure_cache

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"

# DeepSeek-V4-Flash hosted on DeepInfra, routed through LiteLLM/DSPy. No
# reasoning_effort is set, so the provider's default ("normal") effort is used.
# The DeepInfra key is read by LiteLLM from the DEEPINFRA_API_KEY env var at call
# time -- never passed into dspy.LM(...), so it stays out of the OTel trace files.
MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"



class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.lm = dspy.LM(MODEL)
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HoverMultiHop()

    def forward(self, claim):
        self.rm.reset_count()
        with dspy.context(lm=self.lm, rm=self.rm):
            result = self.program(claim=claim)
        result.search_count = self.rm.get_count()
        return result
