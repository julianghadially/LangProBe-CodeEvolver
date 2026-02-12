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

class ExtractFactoidAnswer(dspy.Signature):
    """Extract only the core factoid answer from a raw answer."""

    question = dspy.InputField()
    raw_answer = dspy.InputField()
    answer = dspy.OutputField(desc="only the core factoid answer without any extra descriptive text, parentheticals, or elaboration")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """ChainOfThought variant with full passage reasoning."""

    def __init__(self):
        super().__init__()
        self.create_query_hop2 = dspy.Predict("question,hop1_passages->query")
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.extract_factoid = dspy.Predict(ExtractFactoidAnswer)

    def forward(self, question):
        # HOP 1: Search with the question and scrape top result
        hop1_search_results = self.serper.search(query=question, num_results=1)
        if hop1_search_results:
            hop1_scraped = self.firecrawl.scrape(url=hop1_search_results[0].link)
            if hop1_scraped.success:
                hop1_passages = [hop1_scraped.markdown]
            else:
                # Fallback to snippet if scraping fails
                hop1_passages = [hop1_search_results[0].snippet]
        else:
            hop1_passages = ["No results found for the question."]

        # HOP 2: Generate query, search, and scrape top result
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_passages).query
        hop2_search_results = self.serper.search(query=hop2_query, num_results=1)
        if hop2_search_results:
            hop2_scraped = self.firecrawl.scrape(url=hop2_search_results[0].link)
            if hop2_scraped.success:
                hop2_passages = [hop2_scraped.markdown]
            else:
                # Fallback to snippet if scraping fails
                hop2_passages = [hop2_search_results[0].snippet]
        else:
            hop2_passages = ["No results found for the query."]

        # Generate answer with Chain-of-Thought reasoning over full passages
        answer = self.generate_answer(
            question=question, hop1_passages=hop1_passages, hop2_passages=hop2_passages
        ).answer

        # Extract only the core factoid answer without extra descriptive text
        final_answer = self.extract_factoid(question=question, raw_answer=answer).answer

        return dspy.Prediction(answer=final_answer)
