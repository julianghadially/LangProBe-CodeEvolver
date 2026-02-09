import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HotpotMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Adapted from HoverMultiHop. Hop 3 replaced with answer generation."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")
        self.generate_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")

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
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)


class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict("question,summary_1,summary_2->answer")

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
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
