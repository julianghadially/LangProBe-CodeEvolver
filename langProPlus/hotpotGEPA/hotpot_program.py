import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService
from services.firecrawl_service import FirecrawlService


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="A brief factoid answer matching the expected format - single word, phrase, or yes/no. No explanations or elaborations.")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant using hybrid web search and scraping architecture."""

    def __init__(self):
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.summarize1 = dspy.Predict("question,content->summary")
        self.summarize2 = dspy.Predict("question,context,content->summary")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # HOP 1: Search with original question and scrape top result
        search_results_1 = self.serper.search(query=question, num_results=1)

        if search_results_1:
            top_url_1 = search_results_1[0].link
            scraped_page_1 = self.firecrawl.scrape(url=top_url_1)
            content_1 = scraped_page_1.markdown if scraped_page_1.success else ""
        else:
            content_1 = ""

        summary_1 = self.summarize1(
            question=question, content=content_1
        ).summary

        # HOP 2: Generate refined query and search again, scrape top result
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        search_results_2 = self.serper.search(query=hop2_query, num_results=1)

        if search_results_2:
            top_url_2 = search_results_2[0].link
            scraped_page_2 = self.firecrawl.scrape(url=top_url_2)
            content_2 = scraped_page_2.markdown if scraped_page_2.success else ""
        else:
            content_2 = ""

        summary_2 = self.summarize2(
            question=question, context=summary_1, content=content_2
        ).summary

        # HOP 3: Use ChainOfThought to generate concise extractive answer
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
