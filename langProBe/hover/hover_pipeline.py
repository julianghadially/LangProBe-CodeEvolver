import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class QueryPortfolioGenerator(dspy.Signature):
    """Generate a diverse portfolio of 3-5 candidate search queries using different retrieval strategies.

    Each query should use a different strategy:
    - Entity-focused: queries targeting specific entities (people, places, organizations)
    - Relation-focused: queries emphasizing relationships between entities
    - Temporal-focused: queries incorporating time-related aspects
    - Attribute-focused: queries focusing on properties and characteristics
    - Causal-focused: queries exploring cause-effect relationships

    Generate queries that explore different aspects to maximize retrieval diversity."""

    claim: str = dspy.InputField(desc="the claim to verify")
    context: str = dspy.InputField(desc="summary of previously retrieved information")
    candidate_queries: list[str] = dspy.OutputField(desc="3-5 diverse candidate queries, each using a different retrieval strategy")


class QueryConfidenceScorer(dspy.Signature):
    """Score each candidate query's expected retrieval utility for verifying the claim.

    Evaluate each query based on:
    - Specificity: How precise and targeted is the query?
    - Relevance: How directly does it address the claim?
    - Coverage: Does it fill gaps in prior context?
    - Discriminative power: Will it retrieve distinctive evidence?

    Return a list of scores (0.0-1.0) in the same order as the candidate queries."""

    claim: str = dspy.InputField(desc="the claim to verify")
    context: str = dspy.InputField(desc="summary of previously retrieved information")
    candidate_queries: list[str] = dspy.InputField(desc="the candidate queries to score")
    scores: list[float] = dspy.OutputField(desc="confidence scores (0.0-1.0) for each query, in the same order")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k = 7

        # Adaptive query portfolio modules
        self.query_portfolio_generator = dspy.ChainOfThought(QueryPortfolioGenerator)
        self.query_confidence_scorer = dspy.ChainOfThought(QueryConfidenceScorer)

        # Retrieval and summarization modules
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Generate query portfolio and select best query
            hop1_context = ""  # No prior context for first hop
            hop1_portfolio = self.query_portfolio_generator(
                claim=claim,
                context=hop1_context
            ).candidate_queries

            # Score queries and select top-1
            hop1_scores = self.query_confidence_scorer(
                claim=claim,
                context=hop1_context,
                candidate_queries=hop1_portfolio
            ).scores

            # Select query with highest score
            hop1_best_idx = hop1_scores.index(max(hop1_scores))
            hop1_query = hop1_portfolio[hop1_best_idx]

            # Retrieve using selected query
            hop1_docs = self.retrieve_k(hop1_query).passages
            summary_1 = self.summarize1(
                claim=claim,
                passages=hop1_docs
            ).summary

            # HOP 2: Generate query portfolio and select best query
            hop2_context = summary_1
            hop2_portfolio = self.query_portfolio_generator(
                claim=claim,
                context=hop2_context
            ).candidate_queries

            # Score queries and select top-1
            hop2_scores = self.query_confidence_scorer(
                claim=claim,
                context=hop2_context,
                candidate_queries=hop2_portfolio
            ).scores

            hop2_best_idx = hop2_scores.index(max(hop2_scores))
            hop2_query = hop2_portfolio[hop2_best_idx]

            # Retrieve using selected query
            hop2_docs = self.retrieve_k(hop2_query).passages
            summary_2 = self.summarize2(
                claim=claim,
                context=summary_1,
                passages=hop2_docs
            ).summary

            # HOP 3: Generate query portfolio and select best query
            hop3_context = f"{summary_1}\n{summary_2}"
            hop3_portfolio = self.query_portfolio_generator(
                claim=claim,
                context=hop3_context
            ).candidate_queries

            # Score queries and select top-1
            hop3_scores = self.query_confidence_scorer(
                claim=claim,
                context=hop3_context,
                candidate_queries=hop3_portfolio
            ).scores

            hop3_best_idx = hop3_scores.index(max(hop3_scores))
            hop3_query = hop3_portfolio[hop3_best_idx]

            # Retrieve using selected query
            hop3_docs = self.retrieve_k(hop3_query).passages

            return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
