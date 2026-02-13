import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="Only the minimal factoid answer with NO elaboration, explanation, or additional text. Just the answer itself.")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Two-stage Serper + Firecrawl retrieval for multi-hop reasoning."""

    def __init__(self):
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.create_query_hop1 = dspy.Predict("question->query")
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.summarize1 = dspy.Predict("question,context->summary")
        self.summarize2 = dspy.Predict("question,previous_summary,context->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def _search_and_scrape(self, query: str) -> str:
        """
        Search using Serper and scrape the top Wikipedia result.

        Args:
            query: Search query string

        Returns:
            Markdown content from the scraped page or error message
        """
        try:
            # Search with Serper - focus on Wikipedia results
            search_query = f"{query} site:wikipedia.org"
            results = self.serper.search(search_query, num_results=5)

            if not results:
                return f"No search results found for: {query}"

            # Get the first Wikipedia URL
            top_url = results[0].link

            # Scrape with Firecrawl
            scraped = self.firecrawl.scrape(top_url, max_length=15000)

            if scraped.success:
                return scraped.markdown
            else:
                # Fallback to snippet if scraping fails
                return f"Title: {results[0].title}\n\nSnippet: {results[0].snippet}"

        except Exception as e:
            return f"Error during search/scrape: {str(e)}"

    def forward(self, question):
        # HOP 1: Generate search query for first hop
        hop1_query_obj = self.create_query_hop1(question=question)
        hop1_query = hop1_query_obj.query if hasattr(hop1_query_obj, 'query') else question

        # Search and scrape for first hop
        hop1_context = self._search_and_scrape(hop1_query)

        # Summarize first hop
        summary_1 = self.summarize1(
            question=question,
            context=hop1_context
        ).summary

        # HOP 2: Generate search query for second hop based on first summary
        hop2_query = self.create_query_hop2(
            question=question,
            summary_1=summary_1
        ).query

        # Search and scrape for second hop
        hop2_context = self._search_and_scrape(hop2_query)

        # Summarize second hop with context from first
        summary_2 = self.summarize2(
            question=question,
            previous_summary=summary_1,
            context=hop2_context
        ).summary

        # Generate final answer from both summaries
        answer = self.generate_answer(
            question=question,
            summary_1=summary_1,
            summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
