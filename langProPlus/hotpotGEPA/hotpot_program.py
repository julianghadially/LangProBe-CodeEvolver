import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    hop1_passages = dspy.InputField()
    hop2_passages = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages

        # HOP 2: Use original question for second retrieval
        hop2_docs = self.retrieve_k(question).passages

        # Generate answer using raw passages from both hops
        answer = self.generate_answer(
            question=question, hop1_passages=hop1_docs, hop2_passages=hop2_docs
        ).answer

        return dspy.Prediction(answer=answer)
