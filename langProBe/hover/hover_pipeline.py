import langProBe.hover.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from langProBe.lm_provider import build_task_lm
from .counting_rm import CountingRM
from .hover_program import HoverMultiHop

# Disable DSPy cache so every run exercises the real LM and ColBERT calls.
try:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except AttributeError:
    pass  # older DSPy without configure_cache

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # DeepSeek-V4-Flash (reasoning_effort="high"), served by the preferred
        # provider with a per-call fallback to the same model on the next one
        # down the order when it errors. Provider wiring lives in
        # langProBe/lm_provider.py; $LM_PROVIDER and $LM_FALLBACK repoint the
        # provider / name the cover / disarm the fallback.
        self.lm = build_task_lm()

        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HoverMultiHop()

    def forward(self, claim):
        self.rm.reset_count()
        with dspy.context(lm=self.lm, rm=self.rm):
            result = self.program(claim=claim)
        result.search_count = self.rm.get_count()
        return result
