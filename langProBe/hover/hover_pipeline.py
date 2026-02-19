import dspy
import numpy as np
from sentence_transformers import SentenceTransformer
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractionSignature(dspy.Signature):
    """Extract 2-3 key entities from the claim that are most important for fact-checking."""

    claim: str = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="A list of 2-3 key entities (people, places, organizations, events) from the claim")


class EntityQueryGeneratorSignature(dspy.Signature):
    """Generate a focused search query for a specific entity to find relevant supporting documents."""

    claim: str = dspy.InputField(desc="The original claim")
    entity: str = dspy.InputField(desc="The entity to focus the query on")
    query: str = dspy.OutputField(desc="A focused search query to find documents about this entity in relation to the claim")


class MMRReranker:
    """Implements Maximal Marginal Relevance reranking for document diversity."""

    def __init__(self, lambda_param=0.7):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0). Default 0.7.
        """
        self.lambda_param = lambda_param
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def rerank(self, claim, documents, k=21):
        """
        Rerank documents using MMR to select k diverse and relevant documents.

        Args:
            claim: The original claim
            documents: List of document strings
            k: Number of documents to select

        Returns:
            List of k selected documents
        """
        if len(documents) <= k:
            return documents

        # Encode claim and documents
        claim_embedding = self.encoder.encode([claim], convert_to_numpy=True)[0]
        doc_embeddings = self.encoder.encode(documents, convert_to_numpy=True)

        # Normalize embeddings for cosine similarity
        claim_embedding = claim_embedding / np.linalg.norm(claim_embedding)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Calculate relevance scores (cosine similarity with claim)
        relevance_scores = np.dot(doc_embeddings, claim_embedding)

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Select first document with highest relevance
        first_idx = remaining_indices[np.argmax(relevance_scores[remaining_indices])]
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select k-1 more documents
        while len(selected_indices) < k and remaining_indices:
            selected_embeddings = doc_embeddings[selected_indices]

            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to claim
                relevance = relevance_scores[idx]

                # Max similarity to already selected documents
                similarities = np.dot(selected_embeddings, doc_embeddings[idx])
                max_similarity = np.max(similarities)

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected_doc_idx = remaining_indices[best_idx]
            selected_indices.append(selected_doc_idx)
            remaining_indices.remove(selected_doc_idx)

        # Return selected documents in order
        return [documents[idx] for idx in selected_indices]


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Entity extraction and query generation modules
        self.entity_extractor = dspy.Predict(EntityExtractionSignature)
        self.query_generator = dspy.Predict(EntityQueryGeneratorSignature)

        # High-k retriever for initial retrieval
        self.retrieve_100 = dspy.Retrieve(k=100)

        # MMR reranker for diversity
        self.mmr_reranker = MMRReranker(lambda_param=0.7)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Extract 2-3 key entities from the claim
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities

            # Ensure we have 2-3 entities
            if not isinstance(entities, list):
                entities = [entities]
            entities = entities[:3]  # Limit to max 3 entities

            # Stage 2: Generate focused queries for entities (max 2 queries to stay under 3-query limit)
            # If we have 3 entities, use the first 2; if we have 1-2, use all
            entities_to_query = entities[:2]

            all_documents = []
            seen_docs = set()

            # Retrieve k=100 documents per entity query
            for entity in entities_to_query:
                query_result = self.query_generator(claim=claim, entity=entity)
                query = query_result.query

                # Retrieve 100 documents for this entity query
                retrieved = self.retrieve_100(query)

                # Deduplicate while preserving order
                for doc in retrieved.passages:
                    if doc not in seen_docs:
                        all_documents.append(doc)
                        seen_docs.add(doc)

            # Stage 3: Apply MMR reranking to select final 21 documents
            # MMR maximizes relevance to claim while minimizing similarity to selected docs
            final_documents = self.mmr_reranker.rerank(claim, all_documents, k=21)

            return dspy.Prediction(retrieved_docs=final_documents)
