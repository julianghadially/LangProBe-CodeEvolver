import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.decompose_query = dspy.ChainOfThought("claim -> sub_questions: list[str]")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Decompose the claim into 2-3 specific sub-questions
        decomposition_result = self.decompose_query(claim=claim)
        sub_questions = decomposition_result.sub_questions

        # Ensure we have a list and limit to 3 sub-questions (to respect 3-search limit)
        if isinstance(sub_questions, str):
            # If the output is a string, try to parse it as a list
            sub_questions = [q.strip() for q in sub_questions.split('\n') if q.strip()]

        # Limit to maximum 3 sub-questions
        sub_questions = sub_questions[:3]

        # Retrieve k=7 documents for each sub-question in sequence
        all_docs = []
        for sub_question in sub_questions:
            docs = self.retrieve_k(sub_question).passages
            all_docs.extend(docs)

        return dspy.Prediction(retrieved_docs=all_docs)


