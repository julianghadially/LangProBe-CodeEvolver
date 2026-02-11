import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.refine_answer = dspy.ChainOfThought("question, initial_answer, summary_1, summary_2 -> refined_answer")
        self.refine_answer.__doc__ = "Given an initial answer, refine it to be a SHORT FACTOID ANSWER ONLY. Extract only the most specific entity, name, date, or fact that directly answers the question. Remove all explanatory text, context, and extra information. For person names, include full names with middle names if present in the context."

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        initial_answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        # Refine the initial answer to be a short factoid
        refined = self.refine_answer(
            question=question,
            initial_answer=initial_answer,
            summary_1=summary_1,
            summary_2=summary_2
        ).refined_answer

        return dspy.Prediction(answer=refined)
