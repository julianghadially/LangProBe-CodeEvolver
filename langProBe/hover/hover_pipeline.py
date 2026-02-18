import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop, HoverMultiHopPredict

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


class HoverMultiHopPredictPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop retrieval + prediction pipeline with ColBERTv2.

    This pipeline combines document retrieval with Chain-of-Thought verification
    to predict whether a claim is supported or not supported by retrieved evidence.
    """

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHopPredict()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            return self.program(claim=claim)
