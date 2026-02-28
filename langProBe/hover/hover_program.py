import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class DocumentRelevanceSignature(dspy.Signature):
    """Score a document's relevance to a claim on a scale of 0-10 with justification."""

    claim: str = dspy.InputField(desc="the claim to verify")
    document: str = dspy.InputField(desc="the document to score for relevance")
    justification: str = dspy.OutputField(desc="reasoning for the relevance score")
    score: float = dspy.OutputField(desc="relevance score from 0-10, where 10 is most relevant")


class DocumentReranker(dspy.Module):
    """Re-ranks documents by relevance to a claim using LLM-based scoring."""

    def __init__(self):
        super().__init__()
        self.score_document = dspy.ChainOfThought(DocumentRelevanceSignature)

    def forward(self, claim, documents):
        """Score all documents and return them sorted by relevance score."""
        scored_docs = []

        for doc in documents:
            try:
                result = self.score_document(claim=claim, document=doc)
                # Parse score, handle various formats
                score_str = str(result.score).strip()
                # Extract first number from string (handles "8.5/10" or "8.5" etc)
                score = float(''.join(c for c in score_str.split()[0].split('/')[0] if c.isdigit() or c == '.'))
                scored_docs.append((doc, score))
            except (ValueError, AttributeError):
                # If scoring fails, assign a default low score
                scored_docs.append((doc, 0.0))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return just the documents in sorted order
        return [doc for doc, score in scored_docs]


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k_retrieve = 30  # Over-retrieve 30 documents per hop
        self.k_final = 21  # Final number of documents to return after re-ranking
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k_retrieve)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.reranker = DocumentReranker()

    def forward(self, claim):
        # HOP 1: Over-retrieve with k=30
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Over-retrieve with k=30
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Over-retrieve with k=30
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all retrieved documents (90 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Re-rank documents using LLM-based scoring
        reranked_docs = self.reranker(claim=claim, documents=all_docs)

        # Select top 21 documents
        final_docs = reranked_docs[:self.k_final]

        return dspy.Prediction(retrieved_docs=final_docs)
