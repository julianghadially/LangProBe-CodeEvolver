import dspy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Tuple
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = (
    "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"
)


class EntityExtractor(dspy.Signature):
    """Extract all named entities from the claim to create an entity-focused query."""

    claim: str = dspy.InputField()
    entities: str = dspy.OutputField(
        desc="comma-separated list of all named entities (people, places, organizations, dates, etc.)"
    )
    entity_query: str = dspy.OutputField(
        desc="a search query focused on finding information about these entities"
    )


class RelationshipQueryGenerator(dspy.Signature):
    """Generate a query focused on relationships and connections between entities in the claim."""

    claim: str = dspy.InputField()
    entities: str = dspy.InputField(desc="named entities from the claim")
    relationship_query: str = dspy.OutputField(
        desc="a search query targeting relationships, connections, or interactions between the entities"
    )


class FactVerificationQueryGenerator(dspy.Signature):
    """Generate a query focused on verifying specific claims and facts."""

    claim: str = dspy.InputField()
    verification_query: str = dspy.OutputField(
        desc="a search query to find sources that can verify or refute the specific claims made"
    )


class DocumentScorer(dspy.Signature):
    """Score a document's relevance to the claim on a scale of 0-10 with reasoning."""

    claim: str = dspy.InputField()
    document: str = dspy.InputField()
    reasoning: str = dspy.OutputField(
        desc="explain why this document is or isn't relevant to the claim"
    )
    score: int = dspy.OutputField(
        desc="relevance score from 0 (completely irrelevant) to 10 (highly relevant)"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program."""

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize query generators
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.relationship_query_gen = dspy.Predict(RelationshipQueryGenerator)
        self.fact_verification_query_gen = dspy.Predict(FactVerificationQueryGenerator)

        # Initialize document scorer
        self.document_scorer = dspy.ChainOfThought(DocumentScorer)

        # Initialize sentence transformer for semantic clustering
        self.sentence_encoder = None  # Lazy loading

        # Retrieval parameters
        self.k_per_query = 30
        self.top_scored_docs = 35
        self.final_doc_count = 21
        self.num_clusters = 3
        self.min_docs_per_cluster = 2

    def _get_sentence_encoder(self):
        """Lazy load sentence encoder to avoid initialization overhead."""
        if self.sentence_encoder is None:
            self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.sentence_encoder

    def _retrieve_documents(self, query: str, k: int) -> List[str]:
        """Retrieve k documents for a given query."""
        try:
            results = self.rm(query, k=k)
            return results.passages if hasattr(results, "passages") else results
        except Exception as e:
            print(f"Retrieval error for query '{query}': {e}")
            return []

    def _score_documents(
        self, claim: str, documents: List[str]
    ) -> List[Tuple[str, float]]:
        """Score documents using LLM-based relevance scoring."""
        scored_docs = []
        for doc in documents:
            try:
                result = self.document_scorer(claim=claim, document=doc)
                score = float(result.score)
                scored_docs.append((doc, score))
            except Exception as e:
                # If scoring fails, assign a neutral score
                print(f"Scoring error: {e}")
                scored_docs.append((doc, 5.0))
        return scored_docs

    def _cluster_and_select(self, documents: List[str], target_count: int) -> List[str]:
        """
        Cluster documents by semantic similarity and select diverse representatives.
        Ensures at least min_docs_per_cluster from each cluster.
        """
        if len(documents) <= target_count:
            return documents

        # Get sentence encoder
        encoder = self._get_sentence_encoder()

        # Encode documents
        embeddings = encoder.encode(documents, show_progress_bar=False)

        # Cluster documents
        num_clusters = min(self.num_clusters, len(documents))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Organize documents by cluster
        clusters = {i: [] for i in range(num_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append((idx, documents[idx]))

        # Select documents ensuring diversity
        selected_docs = []
        selected_indices = set()

        # First pass: ensure minimum representation from each cluster
        for cluster_id in range(num_clusters):
            cluster_docs = clusters[cluster_id]
            # Sort by distance to cluster center
            cluster_distances = []
            for idx, doc in cluster_docs:
                dist = np.linalg.norm(
                    embeddings[idx] - kmeans.cluster_centers_[cluster_id]
                )
                cluster_distances.append((idx, doc, dist))
            cluster_distances.sort(key=lambda x: x[2])

            # Take min_docs_per_cluster from each cluster
            for i in range(min(self.min_docs_per_cluster, len(cluster_distances))):
                idx, doc, _ = cluster_distances[i]
                if idx not in selected_indices:
                    selected_docs.append(doc)
                    selected_indices.add(idx)

        # Second pass: fill remaining slots with closest documents to cluster centers
        remaining_slots = target_count - len(selected_docs)
        if remaining_slots > 0:
            all_distances = []
            for cluster_id in range(num_clusters):
                for idx, doc in clusters[cluster_id]:
                    if idx not in selected_indices:
                        dist = np.linalg.norm(
                            embeddings[idx] - kmeans.cluster_centers_[cluster_id]
                        )
                        all_distances.append((idx, doc, dist))

            # Sort by distance and take the closest ones
            all_distances.sort(key=lambda x: x[2])
            for i in range(min(remaining_slots, len(all_distances))):
                idx, doc, _ = all_distances[i]
                selected_docs.append(doc)

        return selected_docs[:target_count]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Generate three complementary queries

            # Query 1: Entity-focused query
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities
            entity_query = entity_result.entity_query

            # Query 2: Relationship-focused query
            relationship_result = self.relationship_query_gen(
                claim=claim, entities=entities
            )
            relationship_query = relationship_result.relationship_query

            # Query 3: Fact-verification query
            fact_result = self.fact_verification_query_gen(claim=claim)
            fact_query = fact_result.verification_query

            # Step 2: Retrieve k=30 documents per query (90 total)
            entity_docs = self._retrieve_documents(entity_query, self.k_per_query)
            relationship_docs = self._retrieve_documents(
                relationship_query, self.k_per_query
            )
            fact_docs = self._retrieve_documents(fact_query, self.k_per_query)

            # Combine and deduplicate documents
            all_docs = entity_docs + relationship_docs + fact_docs
            unique_docs = []
            seen = set()
            for doc in all_docs:
                # Use document content as key for deduplication
                if doc not in seen:
                    seen.add(doc)
                    unique_docs.append(doc)

            # Step 3: LLM-based scoring - score all unique documents
            scored_docs = self._score_documents(claim, unique_docs)

            # Sort by score (descending) and take top 35
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_scored = [doc for doc, score in scored_docs[: self.top_scored_docs]]

            # Step 4: Cluster top 35 documents and select final 21 with diversity
            final_docs = self._cluster_and_select(top_scored, self.final_doc_count)

            return dspy.Prediction(retrieved_docs=final_docs)
