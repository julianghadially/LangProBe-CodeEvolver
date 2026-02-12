import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService
from services.firecrawl_service import FirecrawlService


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    hop1_passages = dspy.InputField()
    hop2_passages = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """ChainOfThought variant with full passage reasoning."""

    def __init__(self):
        super().__init__()
        self.k = 5
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.create_query_hop2 = dspy.Predict("question,hop1_passages->query")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # HOP 1: Search Google with question and scrape top result
        hop1_search_results = self.serper.search(query=question, num_results=self.k)
        hop1_content = ""
        if hop1_search_results:
            top_result = hop1_search_results[0]
            scraped_page = self.firecrawl.scrape(url=top_result.link)
            if scraped_page.success:
                hop1_content = scraped_page.markdown
            else:
                # Fallback to snippet if scraping fails
                hop1_content = top_result.snippet

        # HOP 2: Generate refined query, search again, and scrape top result
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_content).query
        hop2_search_results = self.serper.search(query=hop2_query, num_results=self.k)
        hop2_content = ""
        if hop2_search_results:
            top_result = hop2_search_results[0]
            scraped_page = self.firecrawl.scrape(url=top_result.link)
            if scraped_page.success:
                hop2_content = scraped_page.markdown
            else:
                # Fallback to snippet if scraping fails
                hop2_content = top_result.snippet

        # Generate answer with Chain-of-Thought reasoning over scraped content
        answer = self.generate_answer(
            question=question, hop1_passages=hop1_content, hop2_passages=hop2_content
        ).answer

        return dspy.Prediction(answer=answer)
