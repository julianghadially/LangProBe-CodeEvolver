import dspy
from typing import Optional
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="Only the minimal factoid answer with NO elaboration, explanation, or additional text. Just the answer itself.")


class ExtractFactoid(dspy.Signature):
    """Extract the minimal factoid answer from a detailed response, outputting only 1-5 words that directly answer the question."""

    question = dspy.InputField()
    detailed_answer = dspy.InputField()
    factoid = dspy.OutputField(desc="The minimal factoid answer (1-5 words maximum) that directly answers the question.")


class RerankSearchResults(dspy.Signature):
    """Analyze search results and select the most relevant one for answering a multi-hop question.

    Consider which result is most likely to contain information that helps answer the question,
    taking into account any previous context from earlier reasoning steps.
    """

    question = dspy.InputField(desc="The multi-hop question being answered")
    previous_summary = dspy.InputField(
        desc="Summary from previous reasoning hop (if available). May be empty for first hop."
    )
    search_results = dspy.InputField(
        desc="Search results to rank. Each result starts with [index] and includes Title and Snippet."
    )

    reasoning = dspy.OutputField(
        desc="Step-by-step analysis of which result is most relevant and why"
    )
    selected_index = dspy.OutputField(
        desc="The index (0-based integer) of the most relevant search result"
    )


class SearchResultReranker(dspy.Module):
    """DSPy module for reranking search results based on relevance to multi-hop questions."""

    def __init__(self):
        super().__init__()
        self.reranker = dspy.ChainOfThought(RerankSearchResults)

    def forward(self, question: str, results: list, previous_summary: Optional[str] = None) -> int:
        """Select the best search result.

        Args:
            question: The question being answered
            results: List of SearchResult objects from Serper
            previous_summary: Optional summary from previous hop

        Returns:
            Index of the best result (0-based)
        """
        if not results:
            return 0

        if len(results) == 1:
            return 0

        # Format search results for the LLM
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(
                f"[{i}] Title: {result.title}\n    Snippet: {result.snippet}"
            )
        formatted_str = "\n\n".join(formatted_results)

        # Handle None previous_summary
        summary_text = previous_summary if previous_summary else ""

        # Call the reranker
        try:
            prediction = self.reranker(
                question=question,
                previous_summary=summary_text,
                search_results=formatted_str
            )

            # Extract and validate the selected index
            selected_idx = prediction.selected_index

            # Handle various return types (string, int, etc.)
            if isinstance(selected_idx, str):
                # Try to extract integer from string
                import re
                matches = re.findall(r'\d+', selected_idx)
                if matches:
                    selected_idx = int(matches[0])
                else:
                    selected_idx = 0
            else:
                selected_idx = int(selected_idx)

            # Validate index is within bounds
            if 0 <= selected_idx < len(results):
                return selected_idx
            else:
                return 0

        except Exception as e:
            print(f"Reranker error: {str(e)}. Falling back to first result.")
            return 0


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
        self.extract_factoid = dspy.Predict(ExtractFactoid)
        self.search_reranker = SearchResultReranker()

    def _search_and_scrape(self, query: str, question: Optional[str] = None,
                          previous_summary: Optional[str] = None) -> str:
        """
        Search using Serper, rerank results, and scrape the best result.

        Args:
            query: Search query string
            question: Original question (for reranking context)
            previous_summary: Summary from previous hop (for reranking context)

        Returns:
            Markdown content from the scraped page or error message
        """
        try:
            # Search with Serper - focus on Wikipedia results
            search_query = f"{query} site:wikipedia.org"
            results = self.serper.search(search_query, num_results=5)

            if not results:
                return f"No search results found for: {query}"

            # Rerank results if we have question context, otherwise use first result
            if question:
                best_idx = self.search_reranker(
                    question=question,
                    results=results,
                    previous_summary=previous_summary
                )
                top_url = results[best_idx].link
                print(f"Reranker selected result {best_idx + 1} of {len(results)}")
            else:
                # Fallback to first result if no question context
                best_idx = 0
                top_url = results[0].link

            # Scrape with Firecrawl
            scraped = self.firecrawl.scrape(top_url, max_length=15000)

            if scraped.success:
                return scraped.markdown
            else:
                # Fallback to snippet if scraping fails
                best_idx = best_idx if question else 0
                return f"Title: {results[best_idx].title}\n\nSnippet: {results[best_idx].snippet}"

        except Exception as e:
            return f"Error during search/scrape: {str(e)}"

    def forward(self, question):
        # HOP 1: Generate search query for first hop
        hop1_query_obj = self.create_query_hop1(question=question)
        hop1_query = hop1_query_obj.query if hasattr(hop1_query_obj, 'query') else question

        # Search and scrape for first hop with reranking
        hop1_context = self._search_and_scrape(
            hop1_query,
            question=question,
            previous_summary=None
        )

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

        # Search and scrape for second hop with reranking
        hop2_context = self._search_and_scrape(
            hop2_query,
            question=question,
            previous_summary=summary_1
        )

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

        # Extract concise factoid from the detailed answer
        final_answer = self.extract_factoid(
            question=question,
            detailed_answer=answer
        ).factoid

        return dspy.Prediction(answer=final_answer)
