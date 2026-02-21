import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimDecomposer(dspy.Signature):
    """Decompose the claim into 2-3 focused sub-queries that target different aspects needed for multi-hop reasoning (e.g., main entities, relationships, comparative facts)."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    sub_queries: list[str] = dspy.OutputField(desc="List of 2-3 focused sub-queries that target different aspects needed for multi-hop reasoning")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the claim on a scale from 0 to 10, where 10 is most relevant for verifying the claim through multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    document: str = dspy.InputField(desc="The document to score")
    score: int = dspy.OutputField(desc="Relevance score from 0 to 10, where 10 is most relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize sub-modules for claim decomposition retrieval strategy
        self.initial_retrieve = dspy.Retrieve(k=100)
        self.subquery_retrieve = dspy.Retrieve(k=50)

        # Claim decomposer module
        self.claim_decomposer = dspy.ChainOfThought(ClaimDecomposer)

        # Relevance scorer module (simpler than listwise reranker)
        self.relevance_scorer = dspy.Predict(RelevanceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []

            # Stage 1: Initial retrieval with k=100 documents using the original claim query
            initial_docs = self.initial_retrieve(claim).passages
            all_docs.extend(initial_docs)

            # Stage 2: Decompose claim into 2-3 focused sub-queries
            try:
                decompose_result = self.claim_decomposer(claim=claim)
                sub_queries = decompose_result.sub_queries

                # Limit to exactly 2 sub-queries as specified
                if isinstance(sub_queries, list):
                    sub_queries = sub_queries[:2]
                else:
                    sub_queries = []

                # Stage 3: Retrieve k=50 documents for each sub-query
                for sub_query in sub_queries:
                    try:
                        subquery_docs = self.subquery_retrieve(sub_query).passages
                        all_docs.extend(subquery_docs)
                    except Exception:
                        # If sub-query retrieval fails, continue with next sub-query
                        continue

            except Exception:
                # If claim decomposition fails, continue with just the initial docs
                pass

            # Stage 4: Deduplicate all retrieved documents by normalized title
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0]
                normalized_title = title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Stage 5: Score each document with simple relevance scorer (0-10)
            doc_scores = []
            for idx, doc in enumerate(unique_docs):
                try:
                    score_result = self.relevance_scorer(claim=claim, document=doc)
                    score = score_result.score
                    # Ensure score is an integer between 0 and 10
                    if isinstance(score, (int, float)):
                        score = max(0, min(10, int(score)))
                    else:
                        score = 0
                    doc_scores.append((idx, score))
                except Exception:
                    # If scoring fails, assign a default low score
                    doc_scores.append((idx, 0))

            # Sort by score (descending) and select top 21
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in doc_scores[:21]]

            # Select documents based on scored indices
            final_docs = [unique_docs[idx] for idx in top_indices]

            # Ensure we have exactly up to 21 documents
            final_docs = final_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
