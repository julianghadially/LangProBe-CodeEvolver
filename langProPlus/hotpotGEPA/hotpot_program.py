import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from services import SerperService, SearchResult


class RerankPassagesSignature(dspy.Signature):
    """Rank passages by relevance to the question and select the most useful ones.
    Focus on passages containing specific facts, names, dates, and terminology needed to answer the question."""

    question = dspy.InputField(desc="The question to answer")
    passages = dspy.InputField(desc="Retrieved passages to rank. Each passage starts with [0], [1], [2], etc.")
    top_indices: list[int] = dspy.OutputField(
        desc="Indices of the top 3-5 most relevant passages, ordered by relevance (most relevant first)"
    )


class PassageReranker(LangProBeDSPyMetaProgram, dspy.Module):
    """Rerank passages by relevance and select top-k."""

    def __init__(self, top_k=4):
        super().__init__()
        self.top_k = top_k
        self.ranker = dspy.ChainOfThought(RerankPassagesSignature)

    def forward(self, question, passages):
        """Rerank passages and return top-k most relevant."""
        if not passages or len(passages) == 0:
            return []

        # Format passages with 0-based indices [0], [1], [2], ...
        formatted_passages = self._format_passages(passages)

        # Get ranking from ChainOfThought ranker
        result = self.ranker(question=question, passages=formatted_passages)
        ranking = result.top_indices if hasattr(result, 'top_indices') else []

        # Validate indices and select top passages
        if not ranking or len(ranking) == 0:
            # Fallback: use first top_k passages
            selected = passages[:self.top_k]
        else:
            # Filter valid indices and select top_k
            valid_indices = [idx for idx in ranking if 0 <= idx < len(passages)]
            selected = [passages[idx] for idx in valid_indices[:self.top_k]]

        return selected

    def _format_passages(self, passages):
        """Format passages with [0], [1], [2], ... identifiers (0-based indexing)."""
        formatted = []
        for i, passage in enumerate(passages):
            formatted.append(f"[{i}] {passage}")
        return "\n\n".join(formatted)


def concatenate_contexts(hop1_passages, hop2_passages):
    """Concatenate passages from both hops with clear delineation."""
    context_parts = []

    if hop1_passages:
        context_parts.append("=== First Retrieval ===")
        context_parts.extend(hop1_passages)

    if hop2_passages:
        context_parts.append("=== Second Retrieval ===")
        context_parts.extend(hop2_passages)

    return "\n\n".join(context_parts)


def search_results_to_passages(results: list[SearchResult]) -> list[str]:
    """Convert Serper search results to passage strings.

    Format: Title on first line, snippet on second line.
    This provides clear attribution and maintains context.
    """
    passages = []
    for result in results:
        # Combine title and snippet for richer context
        passage = f"{result.title}\n{result.snippet}"
        passages.append(passage)
    return passages


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved passages that may contain relevant information")
    answer = dspy.OutputField(desc="The answer itself and nothing else")


class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,context->query")
        self.serper = SerperService()
        self.rerank_hop1 = PassageReranker(top_k=4)
        self.rerank_hop2 = PassageReranker(top_k=4)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # HOP 1: Web search and rerank
        hop1_results = self.serper.search(query=question, num_results=self.k)
        hop1_docs = search_results_to_passages(hop1_results)
        reranked_hop1 = self.rerank_hop1(question=question, passages=hop1_docs)

        # Prepare context for hop 2 query generation
        hop1_context = "\n\n".join(reranked_hop1)

        # HOP 2: Generate query using hop1 context, retrieve and rerank
        hop2_query = self.create_query_hop2(
            question=question,
            context=hop1_context
        ).query
        hop2_results = self.serper.search(query=hop2_query, num_results=self.k)
        hop2_docs = search_results_to_passages(hop2_results)
        reranked_hop2 = self.rerank_hop2(question=hop2_query, passages=hop2_docs)

        # Concatenate all reranked passages
        full_context = concatenate_contexts(reranked_hop1, reranked_hop2)

        # Generate answer from concatenated context
        answer = self.generate_answer(
            question=question,
            context=full_context
        ).answer

        return dspy.Prediction(answer=answer)
