import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class PassageRerankerSignature(dspy.Signature):
    """
    Rerank retrieved passages based on their relevance to the claim.
    Prioritize passages containing critical information such as:
    - Entity names mentioned in the claim
    - Dates, numbers, or specific facts referenced in the claim
    - Relationships or connections between entities
    - Evidence that directly supports or refutes the claim

    Return passages in descending order of relevance (most relevant first).
    """

    claim = dspy.InputField(
        desc="The claim to verify. Use this to identify which passages contain the most critical information."
    )

    passages = dspy.InputField(
        desc="Retrieved passages to rerank. Each passage starts with a numerical identifier [N]."
    )

    ranking: list[int] = dspy.OutputField(
        desc="Reranked passage indices in descending order of relevance. Return a list of integers corresponding to passage identifiers, with the most relevant passage first."
    )


def format_passages_for_reranking(passages: list[str]) -> str:
    """
    Format passages with numerical identifiers for reranking.

    Args:
        passages: List of passage strings

    Returns:
        Formatted string with passages numbered [1], [2], etc.
    """
    formatted = []
    for i, passage in enumerate(passages, start=1):
        formatted.append(f"[{i}] {passage}")
    return "\n\n".join(formatted)


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")
        self.reranker = dspy.ChainOfThought(PassageRerankerSignature)

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # RERANKING
        all_passages = hop1_docs + hop2_docs + hop3_docs
        formatted_passages = format_passages_for_reranking(all_passages)
        ranking_indices = self.reranker(
            claim=claim,
            passages=formatted_passages
        ).ranking

        # Validate and handle edge cases
        valid_indices = [idx for idx in ranking_indices if 1 <= idx <= len(all_passages)]

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = [idx for idx in valid_indices if idx not in seen and not seen.add(idx)]

        # Add missing indices at the end
        missing_indices = set(range(1, len(all_passages) + 1)) - set(unique_indices)
        complete_ranking = unique_indices + sorted(list(missing_indices))

        # Convert 1-based indices to 0-based and reorder passages
        reranked_passages = [all_passages[idx - 1] for idx in complete_ranking]

        return dspy.Prediction(retrieved_docs=reranked_passages)


