import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ConfidenceScorerSignature(dspy.Signature):
    """Evaluate the relevance of retrieved passages to the given claim.
    For each passage, assess how well it supports or relates to verifying the claim.
    Use Chain-of-Thought reasoning to determine a confidence score between 0 and 1 for each passage."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    passages: list[str] = dspy.InputField(desc="Retrieved passages to score")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning about passage relevance")
    confidence_scores: list[float] = dspy.OutputField(desc="Confidence score (0-1) for each passage indicating relevance to the claim")


class AdaptiveQueryRefinerSignature(dspy.Signature):
    """Generate an improved query to retrieve more relevant documents for claim verification.
    Analyze the low-confidence passages to understand what information is missing, then create a better query."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    previous_query: str = dspy.InputField(desc="The previous query that was used")
    low_confidence_passages: list[str] = dspy.InputField(desc="Passages that received low confidence scores")
    reasoning: str = dspy.OutputField(desc="Analysis of why previous query failed and what to improve")
    refined_query: str = dspy.OutputField(desc="An improved query to retrieve more relevant documents")


class ListwiseRerankerSignature(dspy.Signature):
    """Rerank all retrieved passages to identify the top 21 most relevant for verifying the claim.
    Consider both the passages content and their confidence scores to select the best evidence."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    passages: list[str] = dspy.InputField(desc="All retrieved passages from all hops")
    confidence_scores: list[float] = dspy.InputField(desc="Confidence score for each passage")
    reasoning: str = dspy.OutputField(desc="Reasoning about which passages are most relevant")
    top_indices: list[int] = dspy.OutputField(desc="Indices of the top 21 most relevant passages (0-indexed)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k = 30
        self.confidence_threshold = 0.6
        self.max_searches = 3

        # Initialize retrieval modules
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Initialize query generation modules for each hop
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

        # Initialize summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Initialize new confidence-based modules
        self.confidence_scorer = dspy.ChainOfThought(ConfidenceScorerSignature)
        self.query_refiner = dspy.ChainOfThought(AdaptiveQueryRefinerSignature)
        self.reranker = dspy.ChainOfThought(ListwiseRerankerSignature)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_passages = []
            all_confidence_scores = []
            search_count = 0

            # HOP 1: Initial retrieval from claim
            hop1_query = claim
            hop1_docs = self.retrieve_k(hop1_query).passages
            search_count += 1

            # Score hop 1 passages
            hop1_scores_result = self.confidence_scorer(claim=claim, passages=hop1_docs)
            hop1_scores = hop1_scores_result.confidence_scores

            # If average confidence is low and we haven't hit max searches, refine and retry
            if len(hop1_scores) > 0 and sum(hop1_scores) / len(hop1_scores) < self.confidence_threshold and search_count < self.max_searches:
                # Get low confidence passages
                low_conf_passages = [doc for doc, score in zip(hop1_docs, hop1_scores) if score < self.confidence_threshold]
                if len(low_conf_passages) > 0:
                    refined_query_result = self.query_refiner(
                        claim=claim,
                        previous_query=hop1_query,
                        low_confidence_passages=low_conf_passages
                    )
                    refined_query = refined_query_result.refined_query
                    additional_docs = self.retrieve_k(refined_query).passages
                    search_count += 1

                    # Score additional docs
                    additional_scores_result = self.confidence_scorer(claim=claim, passages=additional_docs)
                    additional_scores = additional_scores_result.confidence_scores

                    # Merge with hop1
                    hop1_docs = hop1_docs + additional_docs
                    hop1_scores = hop1_scores + additional_scores

            # Accumulate hop1 passages
            all_passages.extend(hop1_docs)
            all_confidence_scores.extend(hop1_scores)

            # Generate summary for hop 1
            summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

            # HOP 2: Query based on claim + summary_1
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = self.retrieve_k(hop2_query).passages
            search_count += 1

            # Score hop 2 passages
            hop2_scores_result = self.confidence_scorer(claim=claim, passages=hop2_docs)
            hop2_scores = hop2_scores_result.confidence_scores

            # If average confidence is low and we haven't hit max searches, refine and retry
            if search_count < self.max_searches and len(hop2_scores) > 0 and sum(hop2_scores) / len(hop2_scores) < self.confidence_threshold:
                low_conf_passages = [doc for doc, score in zip(hop2_docs, hop2_scores) if score < self.confidence_threshold]
                if len(low_conf_passages) > 0:
                    refined_query_result = self.query_refiner(
                        claim=claim,
                        previous_query=hop2_query,
                        low_confidence_passages=low_conf_passages
                    )
                    refined_query = refined_query_result.refined_query
                    additional_docs = self.retrieve_k(refined_query).passages
                    search_count += 1

                    # Score additional docs
                    additional_scores_result = self.confidence_scorer(claim=claim, passages=additional_docs)
                    additional_scores = additional_scores_result.confidence_scores

                    # Merge with hop2
                    hop2_docs = hop2_docs + additional_docs
                    hop2_scores = hop2_scores + additional_scores

            # Accumulate hop2 passages
            all_passages.extend(hop2_docs)
            all_confidence_scores.extend(hop2_scores)

            # Generate summary for hop 2
            summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

            # HOP 3: Query based on claim + summary_1 + summary_2
            hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
            hop3_docs = self.retrieve_k(hop3_query).passages
            search_count += 1

            # Score hop 3 passages
            hop3_scores_result = self.confidence_scorer(claim=claim, passages=hop3_docs)
            hop3_scores = hop3_scores_result.confidence_scores

            # If average confidence is low and we haven't hit max searches, refine and retry
            if search_count < self.max_searches and len(hop3_scores) > 0 and sum(hop3_scores) / len(hop3_scores) < self.confidence_threshold:
                low_conf_passages = [doc for doc, score in zip(hop3_docs, hop3_scores) if score < self.confidence_threshold]
                if len(low_conf_passages) > 0:
                    refined_query_result = self.query_refiner(
                        claim=claim,
                        previous_query=hop3_query,
                        low_confidence_passages=low_conf_passages
                    )
                    refined_query = refined_query_result.refined_query
                    additional_docs = self.retrieve_k(refined_query).passages
                    search_count += 1

                    # Score additional docs
                    additional_scores_result = self.confidence_scorer(claim=claim, passages=additional_docs)
                    additional_scores = additional_scores_result.confidence_scores

                    # Merge with hop3
                    hop3_docs = hop3_docs + additional_docs
                    hop3_scores = hop3_scores + additional_scores

            # Accumulate hop3 passages
            all_passages.extend(hop3_docs)
            all_confidence_scores.extend(hop3_scores)

            # Use ListwiseReranker to select the best 21 documents from the full pool
            rerank_result = self.reranker(
                claim=claim,
                passages=all_passages,
                confidence_scores=all_confidence_scores
            )
            top_indices = rerank_result.top_indices

            # Select the top 21 documents based on reranker output
            # Ensure we have valid indices and cap at 21
            top_indices = [idx for idx in top_indices if 0 <= idx < len(all_passages)][:21]

            # If we don't have enough indices, add remaining by order of confidence
            if len(top_indices) < 21:
                # Get indices not already selected, sorted by confidence
                remaining_indices = [i for i in range(len(all_passages)) if i not in top_indices]
                sorted_remaining = sorted(remaining_indices, key=lambda i: all_confidence_scores[i] if i < len(all_confidence_scores) else 0, reverse=True)
                top_indices.extend(sorted_remaining[:21 - len(top_indices)])

            # Get the final documents
            retrieved_docs = [all_passages[idx] for idx in top_indices]

            return dspy.Prediction(retrieved_docs=retrieved_docs)
