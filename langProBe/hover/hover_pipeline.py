import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
import numpy as np
from collections import Counter
import math

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtraction(dspy.Signature):
    """Extract 2-4 key named entities (people, organizations, places) from the claim that would be critical for fact-checking."""

    claim: str = dspy.InputField(desc="the claim to fact-check")
    entities: list[str] = dspy.OutputField(desc="2-4 key named entities (people, organizations, places) from the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    Uses entity-focused retrieval: extracts entities, retrieves documents for each entity,
    combines with claim-based retrieval, then re-ranks to select top 21 documents.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.entity_extractor = dspy.Predict(EntityExtraction)
        self.entity_retrieve = dspy.Retrieve(k=10)
        self.claim_retrieve = dspy.Retrieve(k=30)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Extract entities from the claim
            extracted = self.entity_extractor(claim=claim)
            entities = extracted.entities if hasattr(extracted, 'entities') else []

            # Limit to 3 entities to stay within search limits
            entities = entities[:3] if len(entities) > 3 else entities

            # Stage 2: Retrieve documents for each entity (k=10 per entity)
            all_passages = []
            for entity in entities:
                entity_results = self.entity_retrieve(entity).passages
                all_passages.extend(entity_results)

            # Stage 3: Retrieve documents for the claim (k=30)
            claim_results = self.claim_retrieve(claim).passages
            all_passages.extend(claim_results)

            # Stage 4: Deduplicate
            deduplicated_passages = deduplicate(all_passages)

            # Stage 5: Re-rank using BM25 to select top 21 documents
            final_docs = self._rerank_with_bm25(claim, deduplicated_passages, top_k=21)

            return dspy.Prediction(retrieved_docs=final_docs)

    def _rerank_with_bm25(self, query: str, passages: list[str], top_k: int = 21) -> list[str]:
        """Re-rank passages using BM25 scoring."""
        if len(passages) <= top_k:
            return passages

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Tokenize query and passages
        query_tokens = query.lower().split()
        tokenized_passages = [passage.lower().split() for passage in passages]

        # Calculate average document length
        avg_doc_len = sum(len(doc) for doc in tokenized_passages) / len(tokenized_passages)

        # Calculate IDF for each query term
        idf = {}
        for term in query_tokens:
            doc_freq = sum(1 for doc in tokenized_passages if term in doc)
            idf[term] = math.log((len(tokenized_passages) - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # Calculate BM25 score for each passage
        scores = []
        for doc in tokenized_passages:
            doc_len = len(doc)
            term_freqs = Counter(doc)
            score = 0.0
            for term in query_tokens:
                if term in term_freqs:
                    tf = term_freqs[term]
                    norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    score += idf.get(term, 0) * norm_tf
            scores.append(score)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return top k passages
        return [passages[i] for i in top_indices]
