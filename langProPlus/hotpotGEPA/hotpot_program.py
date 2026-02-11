import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService


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
        # HOP 1: Search Wikipedia and scrape top result
        hop1_search_query = f"site:wikipedia.org {question}"
        hop1_results = self.serper.search(hop1_search_query, num_results=5)

        if hop1_results:
            hop1_page = self.firecrawl.scrape(hop1_results[0].link)
            hop1_passages = hop1_page.markdown if hop1_page.success else hop1_page.snippet
        else:
            hop1_passages = ""

        summary_1 = self.summarize1(
            question=question, passages=hop1_passages
        ).summary

        # HOP 2: Generate query, search Wikipedia, and scrape top result
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_search_query = f"site:wikipedia.org {hop2_query}"
        hop2_results = self.serper.search(hop2_search_query, num_results=5)

        if hop2_results:
            hop2_page = self.firecrawl.scrape(hop2_results[0].link)
            hop2_passages = hop2_page.markdown if hop2_page.success else hop2_page.snippet
        else:
            hop2_passages = ""

        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_passages
        ).summary

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
