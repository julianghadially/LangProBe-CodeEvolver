import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


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


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved passages that may contain relevant information")
    answer = dspy.OutputField(desc="The answer itself and nothing else")


class RefineAnswer(dspy.Signature):
    """Refine the answer to be concise and match the expected format. Remove unnecessary elaboration while preserving key facts."""

    question = dspy.InputField()
    initial_answer = dspy.InputField()
    refined_answer = dspy.OutputField(desc="The most concise form of the answer that directly answers the question")


class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Predict variant (no ChainOfThought reasoning)."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("question,context->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rerank_hop1 = PassageReranker(top_k=4)
        self.rerank_hop2 = PassageReranker(top_k=4)
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.refine_answer = dspy.ChainOfThought(RefineAnswer)

    def forward(self, question):
        # HOP 1: Retrieve and rerank
        hop1_docs = self.retrieve_k(question).passages
        reranked_hop1 = self.rerank_hop1(question=question, passages=hop1_docs)

        # Prepare context for hop 2 query generation
        hop1_context = "\n\n".join(reranked_hop1)

        # HOP 2: Generate query using hop1 context, retrieve and rerank
        hop2_query = self.create_query_hop2(
            question=question,
            context=hop1_context
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        reranked_hop2 = self.rerank_hop2(question=question, passages=hop2_docs)

        # Concatenate all reranked passages
        full_context = concatenate_contexts(reranked_hop1, reranked_hop2)

        # Generate answer from concatenated context
        initial = self.generate_answer(
            question=question,
            context=full_context
        ).answer
        answer = self.refine_answer(
            question=question,
            initial_answer=initial
        ).refined_answer

        return dspy.Prediction(answer=answer)
