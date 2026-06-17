import langProBe.hover.tracing_setup  # noqa: F401  -- enables DSPy->OTEL spans on import

import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .counting_rm import CountingRM
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-wiki-colbertservice-serve.modal.run/api/search"

# gpt-5.4-nano is a reasoning model but runs with reasoning disabled by default
# (reasoning_effort defaults to "none" -> 0 reasoning tokens). DSPy also fails to
# auto-detect it as a reasoning model because the version dot ("5.4") breaks its
# gpt-5 regex, so we configure it explicitly. "low" is the only non-trivial effort
# this nano model supports (it rejects "minimal"; "medium"/"high" are not honored).
MODEL = "openai/gpt-5.4-nano"



class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.lm = dspy.LM(MODEL, reasoning_effort="low")
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = HoverMultiHop()

    def forward(self, claim):
        self.rm.reset_count()
        with dspy.context(lm=self.lm, rm=self.rm):
            result = self.program(claim=claim)
        result.search_count = self.rm.get_count()
        return result
