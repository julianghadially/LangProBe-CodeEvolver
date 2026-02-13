import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
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
        self.search_web = dspy.ChainOfThought("question->search_query")
        self.serper_service = SerperService()
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.extract_factoid = dspy.Predict(ExtractFactoid)

    def forward(self, question):
        # HOP 1: Web search with Serper.dev
        search_query = self.search_web(question=question).search_query
        search_results = self.serper_service.search(query=search_query, num_results=5)

        # Format Serper results as passages compatible with summarization
        hop1_docs = [
            f"Title: {result.title}\nSnippet: {result.snippet}\nURL: {result.link}"
            for result in search_results
        ]

        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2: Wikipedia ColBERT retrieval with refined query
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        # Extract concise factoid from verbose answer
        factoid = self.extract_factoid(question=question, full_answer=answer).factoid

        return dspy.Prediction(answer=factoid)
