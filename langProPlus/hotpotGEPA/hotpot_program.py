import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class RerankPassages(dspy.Signature):
    """Rerank passages by relevance to the question."""

    question = dspy.InputField()
    passages = dspy.InputField()
    ranked_passages = dspy.OutputField(desc="Top 3 most relevant passages reranked by relevance to the question")


class GenerateAnswer(dspy.Signature):
    """Extract the exact answer span from passages or summaries."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    top_passages = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")


class ExtractFactoid(dspy.Signature):
    """Extract only the essential factoid answer from a verbose answer."""

    question = dspy.InputField()
    full_answer = dspy.InputField()
    factoid = dspy.OutputField(desc='Only the essential factoid answer with no extra words or articles (e.g., "no" not "No, it was not", "2015 until 2017" not "from 2015 to 2017")')

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_hop1 = dspy.Predict(RerankPassages)
        self.rerank_hop2 = dspy.Predict(RerankPassages)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.extract_factoid = dspy.Predict(ExtractFactoid)

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        # Rerank hop1 passages to get top 3 most relevant
        hop1_reranked = self.rerank_hop1(
            question=question, passages=hop1_docs
        ).ranked_passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_reranked
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        # Rerank hop2 passages to get top 3 most relevant
        hop2_reranked = self.rerank_hop2(
            question=question, passages=hop2_docs
        ).ranked_passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_reranked
        ).summary

        # Combine top passages from both hops
        top_passages = hop1_reranked + hop2_reranked

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2, top_passages=top_passages
        ).answer

        # Extract concise factoid from verbose answer
        factoid = self.extract_factoid(question=question, full_answer=answer).factoid

        return dspy.Prediction(answer=factoid)
