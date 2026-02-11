import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class DocumentRelevanceSignature(dspy.Signature):
    """
    Score a document's relevance to a claim on a scale from 0.0 to 1.0.
    Consider whether the document contains information that helps verify, refute, or provide context for the claim.
    A highly relevant document should contain facts directly related to entities, events, or concepts mentioned in the claim.
    """

    claim = dspy.InputField(desc="The claim to fact-check")
    document = dspy.InputField(desc="A retrieved document passage")
    relevance_score: float = dspy.OutputField(
        desc="Relevance score from 0.0 (completely irrelevant) to 1.0 (highly relevant)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) of why this document is relevant or irrelevant to the claim"
    )


class DocumentReranker(LangProBeDSPyMetaProgram, dspy.Module):
    """
    Reranks retrieved documents based on their relevance to a claim.
    Uses ChainOfThought reasoning to score each document individually.
    """

    def __init__(self):
        super().__init__()
        self.reranker = dspy.ChainOfThought(DocumentRelevanceSignature)

    def forward(self, claim, documents, top_k=7):
        """
        Rerank documents by relevance to claim.

        Args:
            claim (str): The claim to check
            documents (list[str]): List of document passages to rerank
            top_k (int): Number of top documents to return

        Returns:
            list[str]: Top k documents sorted by relevance score (highest first)
        """
        if len(documents) == 0:
            return []

        # Score each document individually
        scored_docs = []
        for doc in documents:
            try:
                result = self.reranker(claim=claim, document=doc)
                # Parse and validate score
                score = float(result.relevance_score)
                score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
            except (ValueError, TypeError, AttributeError):
                # Default to 0 if scoring fails
                score = 0.0

            scored_docs.append({
                'document': doc,
                'score': score,
            })

        # Sort by score descending and return top_k documents
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        top_docs = [item['document'] for item in scored_docs[:top_k]]

        return top_docs


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7  # Final number of documents to keep per hop
        self.retrieve_k_initial = 10  # Over-retrieve for reranking
        self.retrieve_initial = dspy.Retrieve(k=self.retrieve_k_initial)
        self.reranker = DocumentReranker()
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1: Retrieve 10, rerank to top 7, then summarize
        hop1_docs_initial = self.retrieve_initial(claim).passages
        hop1_docs = self.reranker(claim=claim, documents=hop1_docs_initial, top_k=self.k)
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2: Generate query, retrieve 10, rerank to top 7, then summarize
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs_initial = self.retrieve_initial(hop2_query).passages
        hop2_docs = self.reranker(claim=claim, documents=hop2_docs_initial, top_k=self.k)
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Generate query, retrieve 10, rerank to top 7 (no summary)
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs_initial = self.retrieve_initial(hop3_query).passages
        hop3_docs = self.reranker(claim=claim, documents=hop3_docs_initial, top_k=self.k)

        # Return all documents: 7 + 7 + 7 = 21 total
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


