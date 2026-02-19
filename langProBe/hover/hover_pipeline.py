import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop, HoverMultiHopCascading

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            return self.program(claim=claim)


class HoverMultiHopCascadingPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Pipeline for HoverMultiHopCascading with ColBERT retrieval.

    Uses 2-stage cascading retrieval architecture:
    - Stage 1: Entity-focused retrieval (7 docs)
    - Stage 2: Relationship-focused 2-hop retrieval (14 docs)

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.
    """

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHopCascading()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            return self.program(claim=claim)
