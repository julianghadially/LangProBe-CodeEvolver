import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate


class ClaimContextRelevanceScorerSignature(dspy.Signature):
    """Score passage relevance to a claim given accumulated context from previous hops.
    Consider both direct relevance and complementary value. Return score 0-100."""

    claim = dspy.InputField(desc="The claim requiring evidence from multiple documents")
    accumulated_context = dspy.InputField(desc="Summary of info from previous retrieval hops")
    passage = dspy.InputField(desc="Retrieved passage to score. Format: 'Key | Text'")
    relevance_score = dspy.OutputField(desc="Score 0-100. Higher = more relevant given context.")


class ClaimContextRelevanceScorer(LangProBeDSPyMetaProgram, dspy.Module):
    """Scores passages for relevance to claim given accumulated context."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ClaimContextRelevanceScorerSignature)

    def forward(self, claim, accumulated_context, passage):
        return self.scorer(claim=claim, accumulated_context=accumulated_context, passage=passage)

    def get_score(self, claim, accumulated_context, passage) -> float:
        """Extract float score from prediction, handling various output formats."""
        result = self.forward(claim, accumulated_context, passage)
        try:
            import re
            score_str = result.relevance_score.strip()
            match = re.search(r'\d+\.?\d*', score_str)
            return float(match.group()) if match else 0.0
        except (ValueError, AttributeError):
            return 0.0


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with adaptive retrieval and reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    RETRIEVAL STRATEGY
    - Adaptive k values: Hop 1=25, Hop 2=20, Hop 3=15 (60 total retrieved)
    - Two-stage reranking: Score all unique documents, select top 21 by relevance
    '''

    def __init__(self):
        super().__init__()
        # Adaptive k values per hop (total 60 docs)
        self.k_hop1 = 25
        self.k_hop2 = 20
        self.k_hop3 = 15

        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve = dspy.Retrieve(k=3)  # Default k, overridden in forward()
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Relevance scorer for two-stage reranking
        self.relevance_scorer = ClaimContextRelevanceScorer()

    def forward(self, claim):
        # HOP 1: Retrieve with k=25
        hop1_docs = self.retrieve(claim, k=self.k_hop1).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs[:7]  # Summarize top 7 for consistency
        ).summary

        # HOP 2: Retrieve with k=20
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve(hop2_query, k=self.k_hop2).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs[:7]
        ).summary

        # HOP 3: Retrieve with k=15
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve(hop3_query, k=self.k_hop3).passages

        # TWO-STAGE RERANKING
        # Stage 1: Deduplicate all 60 retrieved documents
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = deduplicate(all_docs)

        # Stage 2: Score each unique document based on claim + accumulated context
        accumulated_context = f"Summary from hop 1: {summary_1}\nSummary from hop 2: {summary_2}"
        scored_docs = self._score_documents(claim, accumulated_context, unique_docs)

        # Select top 21 highest-scoring documents
        top_docs = self._select_top_k(scored_docs, k=21)

        return dspy.Prediction(retrieved_docs=top_docs)

    def _score_documents(self, claim, accumulated_context, documents):
        """Score all documents for relevance to claim given context."""
        scored = []
        for doc in documents:
            score = self.relevance_scorer.get_score(
                claim=claim,
                accumulated_context=accumulated_context,
                passage=doc
            )
            scored.append((score, doc))
        return scored

    def _select_top_k(self, scored_docs, k=21):
        """Select top k documents by score."""
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs[:k]]
