import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService
from services.firecrawl_service import FirecrawlService


class ExtractAnswer(dspy.Signature):
    """Extract the exact answer from retrieved passages."""

    question = dspy.InputField()
    passages = dspy.InputField()
    answer = dspy.OutputField(desc="extract the exact short factoid answer from the passages - be precise and concise, no elaboration")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Direct passage-to-answer extraction with two-hop retrieval."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,hop1_passages->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_answer = dspy.ChainOfThought(ExtractAnswer)
        self.serper_service = SerperService()
        self.firecrawl_service = FirecrawlService()

    def forward(self, question):
        # HOP 1: Initial retrieval using the question (Wikipedia ColBERT)
        hop1_docs = self.retrieve_k(question).passages

        # HOP 2: Web search + scrape approach
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_docs).query

        # Use Serper web search instead of ColBERT retrieval
        search_results = self.serper_service.search(hop2_query, num_results=5)

        # Scrape the top search result if available
        all_passages = hop1_docs.copy()
        if search_results:
            scraped_page = self.firecrawl_service.scrape(search_results[0].link)
            if scraped_page.success:
                all_passages.append(scraped_page.markdown)

        # Direct extraction: passages -> answer (no intermediate summarization)
        answer = self.extract_answer(
            question=question, passages=all_passages
        ).answer

        return dspy.Prediction(answer=answer)
