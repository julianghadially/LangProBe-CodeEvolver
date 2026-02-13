import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractKeyFacts(dspy.Signature):
    """Extract key facts from summaries to answer the question."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    key_facts = dspy.OutputField(desc="3-5 key factoids as a bulleted list")


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    answer = dspy.OutputField(desc="A single short factoid answer with no articles, qualifiers, or extra words (e.g., \"Hampton Pirates\" not \"The Hampton Pirates\")")


class CreateWebSearchQuery(dspy.Signature):
    """Generate an optimized web search query to find complementary information.

    The query should target specific facts, entities, or relationships that would
    help answer the question based on what we already know from the first search.
    Focus on extracting additional context, recent information, or details not
    found in encyclopedic sources."""

    question = dspy.InputField(desc="The original multi-hop question to answer")
    summary_1 = dspy.InputField(desc="Summary of information from the first search hop")
    query = dspy.OutputField(desc="A concise web search query (3-8 words) targeting complementary information")


class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant with hybrid retrieval (Wikipedia + Web Search)."""

    def __init__(self, serper_service=None):
        """
        Args:
            serper_service: Optional SerperService instance for web search.
                          If None, will attempt to create one at runtime.
        """
        super().__init__()
        self.k = 7  # Number of web search results
        self.k_hop1 = 10  # Number of Wikipedia results (increased)

        # Hop 1: Wikipedia retrieval (ColBERT)
        self.retrieve_k = dspy.Retrieve(k=self.k_hop1)

        # Hop 2: Web search query generation
        self.create_web_query = dspy.Predict(CreateWebSearchQuery)
        self.serper_service = serper_service

        # Summarization and answer generation
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.extract_key_facts = dspy.Predict(ExtractKeyFacts)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def _convert_serper_results_to_passages(self, search_results) -> list[str]:
        """Convert Serper SearchResult objects to passage format.

        Args:
            search_results: List of SearchResult objects from SerperService

        Returns:
            List of passage strings in format "Title: Snippet"
        """
        passages = []
        for result in search_results[:self.k]:  # Limit to k results
            # Format: "Title: Snippet" (consistent with Wikipedia passage format)
            passage = f"{result.title}: {result.snippet}"
            passages.append(passage)
        return passages

    def _web_search(self, query: str) -> list[str]:
        """Perform web search and return passages.

        Args:
            query: Search query string

        Returns:
            List of passage strings, or empty list on error
        """
        if self.serper_service is None:
            # Lazy initialization if not provided
            from services.serper_service import SerperService
            self.serper_service = SerperService()

        try:
            search_results = self.serper_service.search(
                query=query,
                num_results=self.k,
                country="us"
            )
            return self._convert_serper_results_to_passages(search_results)
        except Exception as e:
            # Graceful degradation: log error and return empty passages
            print(f"[WARNING] Web search failed for query '{query}': {e}")
            print("[WARNING] Continuing with empty passages for hop 2")
            return []

    def forward(self, question):
        # HOP 1: Wikipedia retrieval via ColBERT
        hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2: Web search via Serper
        # Generate web search query based on question and first hop summary
        web_query = self.create_web_query(
            question=question,
            summary_1=summary_1
        ).query

        # Execute web search and convert results to passages
        hop2_docs = self._web_search(web_query)

        # Summarize web search results
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # STAGE 1: Extract key facts from summaries
        key_facts = self.extract_key_facts(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).key_facts

        # STAGE 2: Generate answer using ChainOfThought reasoning
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=answer)
