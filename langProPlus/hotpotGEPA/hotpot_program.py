import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractAnswer(dspy.Signature):
    """Extract the exact answer from retrieved passages."""

    question = dspy.InputField()
    passages = dspy.InputField()
    answer = dspy.OutputField(desc="extract the exact short factoid answer from the passages - be precise and concise, no elaboration")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Direct passage-to-answer extraction with two-hop retrieval."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,hop1_passages->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_answer = dspy.ChainOfThought(ExtractAnswer)

    def forward(self, question):
        # HOP 1: Initial retrieval using the question
        hop1_docs = self.retrieve_k(question).passages

        # HOP 2: Refined retrieval using hop1 context
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_docs).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Concatenate all retrieved passages from both hops
        all_passages = hop1_docs + hop2_docs

        # Direct extraction: passages -> answer (no intermediate summarization)
        answer = self.extract_answer(
            question=question, passages=all_passages
        ).answer

        return dspy.Prediction(answer=answer)
