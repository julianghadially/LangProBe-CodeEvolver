import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, FirecrawlService


class GenerateFirstSubQuestion(dspy.Signature):
    """Generate the first sub-question needed to answer a complex multi-hop question."""

    question = dspy.InputField()
    sub_question_1 = dspy.OutputField(desc="First sub-question that needs to be answered to make progress on the main question")


class GenerateSecondSubQuestion(dspy.Signature):
    """Generate the second sub-question using information discovered from answering the first sub-question."""

    question = dspy.InputField()
    sub_answer_1 = dspy.InputField(desc="Answer to the first sub-question, which may contain entities or facts needed for the second question")
    sub_question_2 = dspy.OutputField(desc="Second sub-question that builds on the first answer to complete the multi-hop reasoning")


class AnswerSubQuestion(dspy.Signature):
    """Extract a targeted answer to a specific sub-question from the given context."""

    sub_question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="Specific factual answer to the sub-question based on the context")


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    sub_answer_1 = dspy.InputField()
    sub_answer_2 = dspy.InputField()
    answer = dspy.OutputField(desc="Only the minimal factoid answer with NO elaboration, explanation, or additional text. Just the answer itself.")


class ExtractFactoid(dspy.Signature):
    """Extract the minimal factoid answer from a detailed response, outputting only 1-5 words that directly answer the question."""

    question = dspy.InputField()
    detailed_answer = dspy.InputField()
    factoid = dspy.OutputField(desc="The minimal factoid answer (1-5 words maximum) that directly answers the question.")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Two-stage Serper + Firecrawl retrieval for multi-hop reasoning."""

    def __init__(self):
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.generate_first_sub_question = dspy.Predict(GenerateFirstSubQuestion)
        self.generate_second_sub_question = dspy.Predict(GenerateSecondSubQuestion)
        self.answer_sub_question = dspy.Predict(AnswerSubQuestion)
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.extract_factoid = dspy.Predict(ExtractFactoid)

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
        # Step 1: Generate the first sub-question from the original question only
        first_sub = self.generate_first_sub_question(question=question)
        sub_question_1 = first_sub.sub_question_1

        # Step 2: Search and scrape for first sub-question
        context_1 = self._search_and_scrape(sub_question_1)

        # Step 3: Extract answer to first sub-question from context
        sub_answer_1 = self.answer_sub_question(
            sub_question=sub_question_1,
            context=context_1
        ).answer

        # Step 4: Generate the second sub-question using both the original question
        # AND the first sub-answer as context (enabling multi-hop reasoning)
        second_sub = self.generate_second_sub_question(
            question=question,
            sub_answer_1=sub_answer_1
        )
        sub_question_2 = second_sub.sub_question_2

        # Step 5: Search and scrape for second sub-question
        # (Now the second question can include entities/facts from sub_answer_1)
        context_2 = self._search_and_scrape(sub_question_2)

        # Step 6: Extract answer to second sub-question from context
        sub_answer_2 = self.answer_sub_question(
            sub_question=sub_question_2,
            context=context_2
        ).answer

        # Step 7: Generate final answer from both sub-answers
        answer = self.generate_answer(
            question=question,
            sub_answer_1=sub_answer_1,
            sub_answer_2=sub_answer_2
        ).answer

        # Step 8: Extract concise factoid from the detailed answer
        final_answer = self.extract_factoid(
            question=question,
            detailed_answer=answer
        ).factoid

        return dspy.Prediction(answer=final_answer)
