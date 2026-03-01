import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractor(dspy.Signature):
    """Extract 2-3 distinct entities or key topics from the claim for focused retrieval.

    Each entity should represent a different aspect of the claim that requires separate investigation.
    Entities can be named entities (people, places, organizations) or key concepts/topics."""

    claim: str = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="A list of 2-3 distinct entities or key topics from the claim")


class QueryGenerator(dspy.Signature):
    """Generate a focused search query for a specific entity in the context of the claim."""

    claim: str = dspy.InputField(desc="The original claim being verified")
    entity: str = dspy.InputField(desc="The specific entity or topic to focus on")
    query: str = dspy.OutputField(desc="A focused search query for this entity")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Query decomposition modules
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.query_generator = dspy.Predict(QueryGenerator)

        # Retrieval with higher k (20-25 docs per query)
        self.retrieve_k = dspy.Retrieve(k=22)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract 2-3 entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities

            # Ensure we have 2-3 entities (limit to max 3)
            if isinstance(entities, str):
                # Handle case where entities might be a string
                entities = [e.strip() for e in entities.split(',')]
            entities = entities[:3]  # Limit to max 3 entities

            # Ensure we have at least 1 entity
            if not entities:
                entities = [claim]

            # Step 2: Generate focused queries for each entity
            queries = []
            for entity in entities:
                query_result = self.query_generator(claim=claim, entity=entity)
                queries.append(query_result.query)

            # Step 3: Retrieve documents for each query (20-25 docs per query)
            all_retrieved_docs = []
            for query in queries:
                docs = self.retrieve_k(query).passages
                all_retrieved_docs.extend(docs)

            # Step 4: Re-rank documents
            ranked_docs = self._rerank_documents(
                all_retrieved_docs, claim, entities
            )

            # Step 5: Return top 21 documents
            return dspy.Prediction(retrieved_docs=ranked_docs[:21])

    def _rerank_documents(self, documents, claim, entities):
        """Re-rank documents based on entity mentions, term overlap, and uniqueness.

        Args:
            documents: List of retrieved documents
            claim: The original claim
            entities: List of extracted entities

        Returns:
            List of re-ranked documents (top 21)
        """
        # Normalize claim and entities for matching
        claim_lower = claim.lower()
        claim_terms = set(claim_lower.split())
        entities_lower = [e.lower() for e in entities]

        # Score each document
        doc_scores = []
        seen_docs = set()  # Track unique documents by their title

        for doc in documents:
            doc_lower = doc.lower()

            # Extract document title (before " | " delimiter)
            doc_title = doc.split(" | ")[0] if " | " in doc else doc[:100]

            # Skip duplicate documents
            if doc_title in seen_docs:
                continue
            seen_docs.add(doc_title)

            # Score 1: Number of entities mentioned (0-3 points, weighted heavily)
            entity_count = sum(1 for entity in entities_lower if entity in doc_lower)
            entity_score = entity_count * 3.0

            # Score 2: Term overlap with claim (Jaccard similarity, 0-1 points)
            doc_terms = set(doc_lower.split())
            if doc_terms and claim_terms:
                intersection = len(claim_terms & doc_terms)
                union = len(claim_terms | doc_terms)
                overlap_score = intersection / union if union > 0 else 0.0
            else:
                overlap_score = 0.0

            # Score 3: Document uniqueness bonus (already handled by deduplication)
            # We give a small bonus for being unique (not a duplicate)
            uniqueness_score = 1.0

            # Total score (entity mentions are weighted most heavily)
            total_score = entity_score + overlap_score + uniqueness_score

            doc_scores.append((doc, total_score))

        # Sort by score (descending) and return documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_docs = [doc for doc, score in doc_scores]

        return ranked_docs
