import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class RankDocumentRelevance(dspy.Signature):
    """Analyze a claim and a large pool of retrieved documents, then perform listwise reranking to identify the top 21 most relevant documents.
    Output the indices (0-based) of the 21 most relevant documents in order of relevance, where each index refers to a position in the input documents list."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    documents: list[str] = dspy.InputField(desc="pool of all retrieved documents from multiple hops")
    top_indices: list[int] = dspy.OutputField(desc="list of exactly 21 indices (0-based) corresponding to the most relevant documents, ranked by relevance")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        # Initialize retrieval with k=50 for retrieve-many strategy
        self.retrieve_many = dspy.Retrieve(k=50)
        # Initialize reranker for selecting top 21 from 150 documents
        self.reranker = dspy.ChainOfThought(RankDocumentRelevance)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # RETRIEVE-MANY: Perform 3 hops with k=50 each to gather 150 total documents

            # HOP 1: Initial retrieval based on claim
            hop1_docs = self.retrieve_many(claim).passages

            # HOP 2: Retrieve additional documents using the claim
            # (In a more sophisticated version, this could use gap analysis from hop 1)
            hop2_docs = self.retrieve_many(claim).passages

            # HOP 3: Retrieve final set of documents
            hop3_docs = self.retrieve_many(claim).passages

            # Combine all retrieved documents (150 total)
            all_documents = hop1_docs + hop2_docs + hop3_docs

            # RERANK-TO-21: Use LLM-based listwise reranking to select top 21
            rerank_result = self.reranker(
                claim=claim,
                documents=all_documents
            )

            # Extract top 21 document indices
            top_indices = rerank_result.top_indices

            # Handle case where reranker returns fewer or more than 21 indices
            # Ensure we have exactly 21 unique valid indices
            valid_indices = []
            for idx in top_indices:
                if isinstance(idx, int) and 0 <= idx < len(all_documents):
                    if idx not in valid_indices:
                        valid_indices.append(idx)
                        if len(valid_indices) == 21:
                            break

            # If we don't have 21 indices, fill with remaining documents in order
            if len(valid_indices) < 21:
                for i in range(len(all_documents)):
                    if i not in valid_indices:
                        valid_indices.append(i)
                        if len(valid_indices) == 21:
                            break

            # Select the top 21 documents based on reranked indices
            top_21_docs = [all_documents[idx] for idx in valid_indices[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
