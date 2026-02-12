import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractCandidates(dspy.Signature):
    """Extract 3-5 candidate answer phrases directly from the retrieved passages."""

    question = dspy.InputField()
    passages = dspy.InputField(desc="Combined passages from both retrieval hops")
    candidates = dspy.OutputField(desc="3-5 candidate answer phrases extracted directly from the text, separated by semicolons")


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    candidates = dspy.InputField(desc="Candidate answer phrases extracted from passages")
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Two-module extraction pipeline: first extract candidates from passages, then reason about the best answer."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.extract_candidates = dspy.Predict(ExtractCandidates)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # HOP 1
        self.hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=self.hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        self.hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=self.hop2_docs
        ).summary

        # EXTRACTION: Extract candidate answers from combined passages
        combined_passages = self.hop1_docs + self.hop2_docs
        candidates = self.extract_candidates(
            question=question, passages=combined_passages
        ).candidates

        # HOP 3: Answer selection with chain-of-thought reasoning
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2, candidates=candidates
        ).answer

        return dspy.Prediction(answer=answer)
