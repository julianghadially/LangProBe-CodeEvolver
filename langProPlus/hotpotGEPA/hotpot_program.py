import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class RerankPassages(dspy.Signature):
    """Rerank passages by relevance to the question and return top 3-5 most relevant passages."""

    question = dspy.InputField()
    passages = dspy.InputField(desc="List of retrieved passages to rerank")
    reranked_passages = dspy.OutputField(
        desc="Top 3-5 most relevant passages as a list, reasoning about which contain the most pertinent information for answering the question"
    )


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
        self.rerank = dspy.ChainOfThought(RerankPassages)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        # Rerank hop1 passages to filter top 3-5 most relevant
        reranked_hop1_docs = self.rerank(
            question=question, passages=hop1_docs
        ).reranked_passages
        summary_1 = self.summarize1(
            question=question, passages=reranked_hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        # Rerank hop2 passages to filter top 3-5 most relevant
        reranked_hop2_docs = self.rerank(
            question=question, passages=hop2_docs
        ).reranked_passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=reranked_hop2_docs
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
