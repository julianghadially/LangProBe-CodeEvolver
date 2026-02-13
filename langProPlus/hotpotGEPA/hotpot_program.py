import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService, SearchResult, ScrapedPage


class AnalyzeMissingInfo(dspy.Signature):
    """Analyze scraped content to determine if additional information is needed to answer the question.

    Given a question and content from a web page, determine if the content is sufficient
    to answer the question, or if a follow-up search is needed for missing information.
    """

    question = dspy.InputField(desc="The multi-hop question to answer")
    content = dspy.InputField(desc="Full markdown content from the scraped web page")
    page_title = dspy.InputField(desc="Title of the scraped page")

    needs_more_info = dspy.OutputField(
        desc="'yes' if additional information is needed, 'no' if current content is sufficient"
    )
    missing_aspect = dspy.OutputField(
        desc="Brief description of what specific information is missing (only if needs_more_info is 'yes')"
    )
    refined_query = dspy.OutputField(
        desc="A targeted search query to find the missing information (only if needs_more_info is 'yes')"
    )


class GenerateAnswerFromWeb(dspy.Signature):
    """Generate a factual answer to a multi-hop question using full web page content.

    Use the provided web content to answer the question. The content comes from full
    web pages (not summaries), so all factual details are preserved.
    """

    question = dspy.InputField(desc="The multi-hop question to answer")
    content_1 = dspy.InputField(desc="Full markdown content from the first scraped page")
    page_1_title = dspy.InputField(desc="Title of the first page")
    content_2 = dspy.InputField(
        desc="Full markdown content from the second scraped page (if available, otherwise empty)"
    )
    page_2_title = dspy.InputField(
        desc="Title of the second page (if available, otherwise empty)"
    )

    answer = dspy.OutputField(desc="A short, factual answer to the question")


class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Web search-based multi-hop prediction using SerperService and FirecrawlService."""

    def __init__(self, serper_service=None, firecrawl_service=None):
        """Initialize the web search-based multi-hop predictor.

        Args:
            serper_service: SerperService instance for web search. If None, creates new instance.
            firecrawl_service: FirecrawlService instance for web scraping. If None, creates new instance.
        """
        super().__init__()

        # Initialize services (lazy instantiation if not provided)
        self.serper = serper_service if serper_service is not None else SerperService()
        self.firecrawl = firecrawl_service if firecrawl_service is not None else FirecrawlService()

        # DSPy predictors for LLM operations
        self.analyze_missing_info = dspy.Predict(AnalyzeMissingInfo)
        self.generate_answer = dspy.Predict(GenerateAnswerFromWeb)

        # Configuration
        self.num_search_results = 10  # Get top 10 results per search
        self.max_scrape_length = 10000  # Maximum characters per scraped page

    def forward(self, question):
        """Execute web search-based multi-hop reasoning.

        Flow:
        1. Perform initial web search with original question
        2. Scrape top result
        3. Analyze if more information is needed
        4. Optionally perform second targeted search and scrape
        5. Generate answer from full content(s)

        Args:
            question: The multi-hop question to answer

        Returns:
            dspy.Prediction with answer field
        """

        # SEARCH 1: Initial search with original question
        print(f"[Search 1] Searching for: {question}")
        try:
            search_results_1 = self.serper.search(
                query=question,
                num_results=self.num_search_results
            )
        except Exception as e:
            print(f"[Search 1 Error] Failed: {str(e)}")
            # Fallback: return error message
            return dspy.Prediction(
                answer=f"Error: Unable to perform web search - {str(e)}"
            )

        # Validate search results
        if not search_results_1 or len(search_results_1) == 0:
            print("[Search 1 Error] No search results found")
            return dspy.Prediction(answer="Error: No search results found")

        # SCRAPE 1: Scrape the top result
        top_result_1 = search_results_1[0]
        print(f"[Scrape 1] Scraping: {top_result_1.title} ({top_result_1.link})")

        scraped_page_1 = self.firecrawl.scrape(
            url=top_result_1.link,
            max_length=self.max_scrape_length,
            skip_pdfs=True
        )

        # Handle scraping failure for first page
        if not scraped_page_1.success:
            print(f"[Scrape 1 Error] Failed to scrape first page: {scraped_page_1.error}")
            # Try second result as fallback
            if len(search_results_1) > 1:
                fallback_result = search_results_1[1]
                print(f"[Scrape 1 Fallback] Trying: {fallback_result.title}")
                scraped_page_1 = self.firecrawl.scrape(
                    url=fallback_result.link,
                    max_length=self.max_scrape_length,
                    skip_pdfs=True
                )
                if not scraped_page_1.success:
                    return dspy.Prediction(
                        answer="Error: Unable to scrape any search results"
                    )
            else:
                return dspy.Prediction(
                    answer="Error: Unable to scrape search result"
                )

        content_1 = scraped_page_1.markdown
        page_1_title = scraped_page_1.title or top_result_1.title

        print(f"[Scrape 1 Success] Retrieved {len(content_1)} characters")

        # ANALYZE: Determine if second search is needed
        print("[Analyze] Checking if additional information is needed...")
        analysis = self.analyze_missing_info(
            question=question,
            content=content_1,
            page_title=page_1_title
        )

        needs_second_search = analysis.needs_more_info.strip().lower() == "yes"

        # Initialize second content as empty (may be populated below)
        content_2 = ""
        page_2_title = ""

        # SEARCH 2: Conditional second search if needed
        if needs_second_search:
            print(f"[Search 2] Additional info needed: {analysis.missing_aspect}")
            print(f"[Search 2] Refined query: {analysis.refined_query}")

            try:
                search_results_2 = self.serper.search(
                    query=analysis.refined_query,
                    num_results=self.num_search_results
                )

                if search_results_2 and len(search_results_2) > 0:
                    # SCRAPE 2: Scrape top result from second search
                    top_result_2 = search_results_2[0]
                    print(f"[Scrape 2] Scraping: {top_result_2.title} ({top_result_2.link})")

                    scraped_page_2 = self.firecrawl.scrape(
                        url=top_result_2.link,
                        max_length=self.max_scrape_length,
                        skip_pdfs=True
                    )

                    if scraped_page_2.success:
                        content_2 = scraped_page_2.markdown
                        page_2_title = scraped_page_2.title or top_result_2.title
                        print(f"[Scrape 2 Success] Retrieved {len(content_2)} characters")
                    else:
                        print(f"[Scrape 2 Error] Failed: {scraped_page_2.error}")
                        # Continue with just first page content
                else:
                    print("[Search 2 Error] No results for refined query")
                    # Continue with just first page content

            except Exception as e:
                print(f"[Search 2 Error] Failed: {str(e)}")
                # Continue with just first page content
        else:
            print("[Analyze] First page content is sufficient")

        # GENERATE ANSWER: Use full content from scraped pages
        print("[Generate] Creating final answer from web content...")
        answer = self.generate_answer(
            question=question,
            content_1=content_1,
            page_1_title=page_1_title,
            content_2=content_2,
            page_2_title=page_2_title
        ).answer

        print(f"[Complete] Answer: {answer}")

        return dspy.Prediction(answer=answer)
