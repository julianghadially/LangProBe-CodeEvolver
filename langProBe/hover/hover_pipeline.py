import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DocumentRelevanceVerifier(dspy.Signature):
    """Evaluate whether the retrieved documents contain relevant information for verifying the claim.
    Assess if the current set of documents is sufficient or if additional retrieval with a refined query would be beneficial."""

    claim: str = dspy.InputField(desc="The claim to verify")
    retrieved_documents: str = dspy.InputField(desc="The documents retrieved so far")
    relevance_score: float = dspy.OutputField(desc="A score from 0.0 to 1.0 indicating overall relevance of documents to the claim")
    is_sufficient: bool = dspy.OutputField(desc="Whether the retrieved documents are sufficient to verify the claim (True) or if more retrieval is needed (False)")
    suggested_refinement: str = dspy.OutputField(desc="If not sufficient, suggest a refined query to retrieve better documents")


class DocumentRanker(dspy.Signature):
    """Rank all retrieved documents by their relevance to the claim. Output a list of document indices ordered by relevance (most relevant first)."""

    claim: str = dspy.InputField(desc="The claim to verify")
    documents: str = dspy.InputField(desc="All retrieved documents numbered [1], [2], etc. to rank")
    ranked_indices: list[int] = dspy.OutputField(desc="List of document numbers ordered by relevance, e.g., [3, 1, 5, 2, 4] means document 3 is most relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using self-critique iterative retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Uses iterative retrieval with self-critique verification (max 3 retrievals)
    - Each retrieval fetches k=10 documents
    - After each retrieval, a verifier assesses document relevance and sufficiency
    - If insufficient, generates a refined query for next iteration
    - Finally ranks all retrieved documents and returns top 21'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.retrieve_k10 = dspy.Retrieve(k=10)
        self.verifier = dspy.ChainOfThought(DocumentRelevanceVerifier)
        self.ranker = dspy.Predict(DocumentRanker)
        self.max_iterations = 2  # Maximum refinement iterations (total 3 retrievals)
        self.max_retrievals = 3  # Total number of retrievals allowed

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_documents = []
            query = claim
            retrieval_count = 0

            # Initial retrieval
            initial_docs = self.retrieve_k10(query).passages
            all_documents.extend(initial_docs)
            retrieval_count += 1

            # Self-critique iterative retrieval loop (max 2 iterations)
            for iteration in range(self.max_iterations):
                # Check if we've hit the retrieval limit
                if retrieval_count >= self.max_retrievals:
                    break

                # Format documents for verification
                doc_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(all_documents)])

                # Verify document relevance and sufficiency
                verification = self.verifier(
                    claim=claim,
                    retrieved_documents=doc_text
                )

                # If documents are sufficient, exit the loop
                if verification.is_sufficient:
                    break

                # Generate refined query and retrieve more documents
                refined_query = verification.suggested_refinement
                new_docs = self.retrieve_k10(refined_query).passages
                all_documents.extend(new_docs)
                retrieval_count += 1

            # Deduplicate documents while preserving order
            seen = set()
            unique_documents = []
            for doc in all_documents:
                if doc not in seen:
                    seen.add(doc)
                    unique_documents.append(doc)

            # Rank all unique documents by relevance
            if len(unique_documents) > 21:
                doc_text = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(unique_documents)])
                ranking_result = self.ranker(claim=claim, documents=doc_text)

                # Extract ranked documents using indices
                try:
                    if isinstance(ranking_result.ranked_indices, list) and len(ranking_result.ranked_indices) > 0:
                        ranked_docs = []
                        for idx in ranking_result.ranked_indices[:21]:
                            # Convert 1-indexed to 0-indexed and validate
                            if isinstance(idx, int) and 1 <= idx <= len(unique_documents):
                                ranked_docs.append(unique_documents[idx - 1])
                        # If we didn't get enough valid indices, fill with remaining docs
                        if len(ranked_docs) < 21:
                            for doc in unique_documents:
                                if doc not in ranked_docs and len(ranked_docs) < 21:
                                    ranked_docs.append(doc)
                    else:
                        # Fallback if ranking didn't work as expected
                        ranked_docs = unique_documents[:21]
                except (AttributeError, IndexError, TypeError):
                    # Fallback on any error
                    ranked_docs = unique_documents[:21]
            else:
                ranked_docs = unique_documents

            return dspy.Prediction(retrieved_docs=ranked_docs)
