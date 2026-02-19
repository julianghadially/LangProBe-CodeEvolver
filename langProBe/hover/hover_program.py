import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ContrastiveQuerySignature(dspy.Signature):
    """Generate both a positive query for relevant information and a negative query
    representing what NOT to retrieve based on irrelevant documents already seen.

    The positive query should target missing information gaps.
    The negative query should represent patterns, topics, or content from irrelevant documents
    that should be avoided in future retrievals."""

    claim = dspy.InputField(desc="The claim to verify")
    previous_summary = dspy.InputField(desc="Summary of previously retrieved documents")
    retrieved_passages = dspy.InputField(desc="Recently retrieved passages to analyze")
    cumulative_negative_context = dspy.InputField(
        desc="Accumulated context of what NOT to retrieve from previous hops",
        default=""
    )

    positive_query = dspy.OutputField(
        desc="Query targeting missing information gaps that would help verify the claim"
    )
    negative_query = dspy.OutputField(
        desc="Query representing irrelevant content patterns to avoid (topics, keywords, or themes from unhelpful documents)"
    )


class ContrastiveQuerySignatureHop2(dspy.Signature):
    """Generate both a positive query for relevant information and a negative query
    for hop 2 based on hop 1 results."""

    claim = dspy.InputField(desc="The claim to verify")
    summary_1 = dspy.InputField(desc="Summary of hop 1 documents")

    positive_query = dspy.OutputField(
        desc="Query targeting missing information gaps that would help verify the claim"
    )
    negative_query = dspy.OutputField(
        desc="Query representing irrelevant content patterns to avoid based on hop 1"
    )


class ContrastiveQuerySignatureHop3(dspy.Signature):
    """Generate both a positive query for relevant information and a negative query
    for hop 3 based on hop 1 and 2 results."""

    claim = dspy.InputField(desc="The claim to verify")
    summary_1 = dspy.InputField(desc="Summary of hop 1 documents")
    summary_2 = dspy.InputField(desc="Summary of hop 2 documents")
    cumulative_negative_context = dspy.InputField(
        desc="Accumulated negative queries from previous hops"
    )

    positive_query = dspy.OutputField(
        desc="Query targeting missing information gaps that would help verify the claim"
    )
    negative_query = dspy.OutputField(
        desc="Query representing irrelevant content patterns to avoid based on hops 1 and 2"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with contrastive query learning.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Uses ContrastiveQueryGenerator to create positive and negative queries after each hop
    - Retrieves k=15 documents per hop, then reranks using contrast scoring
    - Selects top 7 documents per hop based on weighted scoring:
      * Positive similarity: how well document matches positive query
      * Negative dissimilarity: how different document is from negative query
    - Tracks cumulative negative context across hops to avoid repeated mistakes
    '''

    def __init__(self, alpha=0.6, beta=0.4):
        """
        Args:
            alpha: Weight for positive query similarity (default: 0.6)
            beta: Weight for negative query dissimilarity (default: 0.4)
        """
        super().__init__()
        self.k_retrieve = 15  # Retrieve more documents for reranking
        self.k_final = 7      # Final number of documents to keep
        self.alpha = alpha    # Weight for positive similarity
        self.beta = beta      # Weight for negative dissimilarity

        # Contrastive query generators for each hop
        self.create_query_hop2 = dspy.ChainOfThought(ContrastiveQuerySignatureHop2)
        self.create_query_hop3 = dspy.ChainOfThought(ContrastiveQuerySignatureHop3)

        # Retriever with larger k for reranking
        self.retrieve_k = dspy.Retrieve(k=self.k_retrieve)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Track negative queries across hops
        self.negative_queries_history = []

    def compute_contrast_score(self, doc_text, positive_query, negative_query):
        """
        Compute contrast score for a document based on positive and negative queries.

        Simple heuristic: count matching terms (in practice, this would use embeddings).
        Higher score = better match to positive query and worse match to negative query.

        Args:
            doc_text: Document text
            positive_query: Query for relevant content
            negative_query: Query for irrelevant content

        Returns:
            float: Weighted contrast score
        """
        # Convert to lowercase for matching
        doc_lower = doc_text.lower()
        pos_terms = set(positive_query.lower().split())
        neg_terms = set(negative_query.lower().split())
        doc_terms = set(doc_lower.split())

        # Positive similarity: proportion of positive query terms in document
        pos_overlap = len(pos_terms & doc_terms)
        pos_score = pos_overlap / max(len(pos_terms), 1)

        # Negative dissimilarity: inverse of proportion of negative query terms in document
        neg_overlap = len(neg_terms & doc_terms)
        neg_score = 1.0 - (neg_overlap / max(len(neg_terms), 1))

        # Weighted combination
        contrast_score = self.alpha * pos_score + self.beta * neg_score

        return contrast_score

    def rerank_with_contrast(self, documents, positive_query, negative_query):
        """
        Rerank documents using contrastive scoring.

        Args:
            documents: List of document strings (length k_retrieve=15)
            positive_query: Query targeting relevant information
            negative_query: Query representing irrelevant content

        Returns:
            list: Top k_final (7) documents after contrastive reranking
        """
        if not documents:
            return []

        # Score each document
        scored_docs = []
        for doc in documents:
            score = self.compute_contrast_score(doc, positive_query, negative_query)
            scored_docs.append((score, doc))

        # Sort by score (descending) and take top k_final
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        reranked_docs = [doc for score, doc in scored_docs[:self.k_final]]

        return reranked_docs

    def forward(self, claim):
        """
        Forward pass with contrastive query generation and reranking.

        Process:
        1. HOP 1: Retrieve with claim directly (no contrast yet)
        2. HOP 2: Generate contrastive queries, retrieve k=15, rerank to top 7
        3. HOP 3: Generate contrastive queries with cumulative negative context,
                  retrieve k=15, rerank to top 7

        Total: 7 + 7 + 7 = 21 documents (at most)
        """
        # Reset negative queries history for this forward pass
        self.negative_queries_history = []

        # ============ HOP 1 ============
        # Initial retrieval with the claim (no contrast learning yet)
        hop1_docs_full = self.retrieve_k(claim).passages
        # Take only top k_final for hop 1
        hop1_docs = hop1_docs_full[:self.k_final]

        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # ============ HOP 2 ============
        # Generate contrastive queries based on hop 1
        hop2_queries = self.create_query_hop2(
            claim=claim,
            summary_1=summary_1
        )
        positive_query_2 = hop2_queries.positive_query
        negative_query_2 = hop2_queries.negative_query

        # Store negative query for cumulative context
        self.negative_queries_history.append(negative_query_2)

        # Retrieve k=15 documents with positive query
        hop2_docs_full = self.retrieve_k(positive_query_2).passages

        # Rerank using contrast scoring and select top 7
        hop2_docs = self.rerank_with_contrast(
            hop2_docs_full,
            positive_query_2,
            negative_query_2
        )

        summary_2 = self.summarize2(
            claim=claim,
            context=summary_1,
            passages=hop2_docs
        ).summary

        # ============ HOP 3 ============
        # Build cumulative negative context from all previous hops
        cumulative_negative_context = " | ".join(self.negative_queries_history)

        # Generate contrastive queries for hop 3
        hop3_queries = self.create_query_hop3(
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            cumulative_negative_context=cumulative_negative_context
        )
        positive_query_3 = hop3_queries.positive_query
        negative_query_3 = hop3_queries.negative_query

        # Store negative query
        self.negative_queries_history.append(negative_query_3)

        # Retrieve k=15 documents with positive query
        hop3_docs_full = self.retrieve_k(positive_query_3).passages

        # Rerank using contrast scoring and select top 7
        hop3_docs = self.rerank_with_contrast(
            hop3_docs_full,
            positive_query_3,
            negative_query_3
        )

        # Return all retrieved documents (7 + 7 + 7 = 21 max)
        return dspy.Prediction(
            retrieved_docs=hop1_docs + hop2_docs + hop3_docs,
            negative_queries=self.negative_queries_history,
            positive_queries=[positive_query_2, positive_query_3]
        )
