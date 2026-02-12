import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateNormalizedAnswer(dspy.Signature):
    """Answer questions with a normalized factoid answer that preserves essential qualifiers."""

    question = dspy.InputField()
    hop1_passages = dspy.InputField()
    hop2_passages = dspy.InputField()
    answer = dspy.OutputField(desc="A normalized answer that preserves essential qualifiers like titles (e.g., 'King', 'President'), full names with middle names and suffixes (e.g., 'Jr.', 'Sr.', 'III'), and exact terminology. Provide only the answer itself without extra descriptive text or elaboration.")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """ChainOfThought variant with full passage reasoning."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,hop1_passages->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.generate_answer = dspy.ChainOfThought(GenerateNormalizedAnswer)

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_docs).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Generate normalized answer with Chain-of-Thought reasoning over full passages
        answer = self.generate_answer(
            question=question, hop1_passages=hop1_docs, hop2_passages=hop2_docs
        ).answer

        return dspy.Prediction(answer=answer)
