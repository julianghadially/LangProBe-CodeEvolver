import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services.serper_service import SerperService


class RerankPassages(dspy.Signature):
    """Rerank passages by relevance to the question."""

    question = dspy.InputField()
    passages = dspy.InputField()
    ranked_passages = dspy.OutputField(desc="Top 3 most relevant passages reranked by relevance to the question")


class GenerateAnswer(dspy.Signature):
    """Extract the exact answer span from passages or summaries."""

    question = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    top_passages = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")


class RefineAnswerWithWebContext(dspy.Signature):
    """Refine the initial answer using web search results to provide complete, authoritative details in the expected format."""

    question = dspy.InputField()
    initial_answer = dspy.InputField()
    summary_1 = dspy.InputField()
    summary_2 = dspy.InputField()
    web_search_results = dspy.InputField()
    refined_answer = dspy.OutputField(desc="The complete, authoritative answer with full details (e.g., full names, complete locations) verified against web search results. Match the expected format exactly.")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,summary_1->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_hop1 = dspy.Predict(RerankPassages)
        self.rerank_hop2 = dspy.Predict(RerankPassages)
        self.summarize1 = dspy.Predict("question,passages->summary")
        self.summarize2 = dspy.Predict("question,context,passages->summary")
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.refine_with_web = dspy.Predict(RefineAnswerWithWebContext)
        self.serper_service = SerperService()

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        # Rerank hop1 passages to get top 3 most relevant
        hop1_reranked = self.rerank_hop1(
            question=question, passages=hop1_docs
        ).ranked_passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_reranked
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        # Rerank hop2 passages to get top 3 most relevant
        hop2_reranked = self.rerank_hop2(
            question=question, passages=hop2_docs
        ).ranked_passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_reranked
        ).summary

        # Combine top passages from both hops
        top_passages = hop1_reranked + hop2_reranked

        # HOP 3: Answer instead of another query+retrieve
        answer = self.generate_answer(
            question=question, summary_1=summary_1, summary_2=summary_2, top_passages=top_passages
        ).answer

        # Web search verification: Use web search to find complete, authoritative answer format
        # Construct a search query combining question and initial answer for verification
        web_query = f"{question} {answer}"
        web_results = self.serper_service.search(query=web_query, num_results=5)

        # Format web search results as text passages
        web_search_text = "\n\n".join([
            f"[{i+1}] {result.title}\n{result.snippet}"
            for i, result in enumerate(web_results)
        ])

        # Refine answer using web context to get complete details (full names, complete locations)
        refined_answer = self.refine_with_web(
            question=question,
            initial_answer=answer,
            summary_1=summary_1,
            summary_2=summary_2,
            web_search_results=web_search_text
        ).refined_answer

        return dspy.Prediction(answer=refined_answer)
