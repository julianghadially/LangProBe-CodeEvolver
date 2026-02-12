import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService
from services.firecrawl_service import FirecrawlService


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
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question):
        # HOP 1: Search and scrape
        hop1_results = self.serper.search(question, num_results=1)
        if hop1_results:
            hop1_scraped = self.firecrawl.scrape(hop1_results[0].link)
            hop1_docs = hop1_scraped.markdown if hop1_scraped.success else hop1_results[0].snippet
        else:
            hop1_docs = ""

        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2: Generate query, search and scrape
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_results = self.serper.search(hop2_query, num_results=1)
        if hop2_results:
            hop2_scraped = self.firecrawl.scrape(hop2_results[0].link)
            hop2_docs = hop2_scraped.markdown if hop2_scraped.success else hop2_results[0].snippet
        else:
            hop2_docs = ""

        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
