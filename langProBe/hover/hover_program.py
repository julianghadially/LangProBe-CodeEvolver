import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from typing import List, Dict, Tuple
import re


class EntityExtractor(dspy.Signature):
    """Extract key entities (proper nouns, titles, specific references) from the claim that are essential for fact verification.
    Focus on named entities like people, places, organizations, titles, dates, and specific terms that require verification."""

    claim = dspy.InputField()
    entities: List[str] = dspy.OutputField(desc="List of 1-3 key entities to verify, ordered by importance")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused search query for a specific entity in the context of the claim.
    The query should help retrieve documents that contain information about this entity relevant to the claim."""

    claim = dspy.InputField()
    entity = dspy.InputField()
    query = dspy.OutputField(desc="A search query focused on this entity in the claim's context")


class DocumentReranker:
    """Reranks documents based on entity presence, title matching, and relevance to the claim."""

    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def extract_title(self, doc: str) -> str:
        """Extract document title (assumes format 'Title | Content')."""
        if ' | ' in doc:
            return doc.split(' | ')[0]
        return doc[:100]  # Fallback to first 100 chars

    def score_document(self, doc: str, claim: str, entities: List[str]) -> float:
        """Score a document based on multiple relevance factors."""
        doc_normalized = self.normalize_text(doc)
        claim_normalized = self.normalize_text(claim)
        title = self.extract_title(doc)
        title_normalized = self.normalize_text(title)

        score = 0.0

        # Score 1: Entity presence (most important)
        entity_count = 0
        for entity in entities:
            entity_normalized = self.normalize_text(entity)
            if entity_normalized in doc_normalized:
                entity_count += 1
            # Bonus for entity in title
            if entity_normalized in title_normalized:
                entity_count += 0.5

        # Normalize entity score: 0-50 points
        if entities:
            score += (entity_count / len(entities)) * 50

        # Score 2: Title/key term matching (30 points)
        # Extract key terms from claim (words longer than 3 chars, excluding common words)
        common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'has', 'was', 'were', 'are', 'been'}
        claim_words = [w for w in claim_normalized.split() if len(w) > 3 and w not in common_words]

        if claim_words:
            title_matches = sum(1 for word in claim_words if word in title_normalized)
            score += (title_matches / len(claim_words)) * 30

        # Score 3: Semantic relevance via word overlap (20 points)
        doc_words = set(doc_normalized.split()[:200])  # Use first 200 words for efficiency
        claim_words_set = set(claim_words)

        if claim_words_set:
            overlap = len(doc_words.intersection(claim_words_set))
            score += min(overlap / len(claim_words_set), 1.0) * 20

        return score

    def rerank(self, documents: List[str], claim: str, entities: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents and return list of (document, score) tuples."""
        scored_docs = []
        for doc in documents:
            score = self.score_document(doc, claim, entities)
            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverEntityAware(LangProBeDSPyMetaProgram, dspy.Module):
    '''Entity-aware multi-hop system for retrieving documents for a provided claim.

    This system:
    1. Extracts key entities from the claim
    2. Generates entity-specific queries (max 3)
    3. Retrieves k=30 documents per entity query
    4. Reranks documents based on entity presence, title matching, and semantic relevance
    5. Deduplicates documents
    6. Selects top 21 documents

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Entity extraction and query generation
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)
        self.entity_query_generator = dspy.ChainOfThought(EntityQueryGenerator)

        # Retrieval with k=30 per entity query
        self.retrieve_k30 = dspy.Retrieve(k=30)

        # Reranker
        self.reranker = DocumentReranker()

        # Max number of entity queries to respect constraints
        self.max_entity_queries = 3

        # Target number of final documents
        self.target_docs = 21

    def deduplicate_documents(self, documents: List[str]) -> List[str]:
        """Remove duplicate documents while preserving order."""
        seen = set()
        unique_docs = []

        for doc in documents:
            # Normalize for comparison
            doc_normalized = self.reranker.normalize_text(doc)

            if doc_normalized not in seen:
                seen.add(doc_normalized)
                unique_docs.append(doc)

        return unique_docs

    def forward(self, claim):
        # Step 1: Extract key entities from the claim
        entity_result = self.entity_extractor(claim=claim)
        entities = entity_result.entities

        # Limit to max 3 entities to respect constraint
        if len(entities) > self.max_entity_queries:
            entities = entities[:self.max_entity_queries]

        # Fallback: if no entities extracted, use claim as single query
        if not entities:
            entities = [claim]

        # Step 2 & 3: Generate entity-specific queries and retrieve k=30 documents per query
        all_retrieved_docs = []

        for entity in entities:
            # Generate entity-specific query
            query_result = self.entity_query_generator(claim=claim, entity=entity)
            entity_query = query_result.query

            # Retrieve k=30 documents for this entity query
            retrieved = self.retrieve_k30(entity_query).passages
            all_retrieved_docs.extend(retrieved)

        # Step 4: Deduplicate documents (before reranking to save computation)
        unique_docs = self.deduplicate_documents(all_retrieved_docs)

        # Step 5: Rerank documents based on relevance
        reranked_docs = self.reranker.rerank(unique_docs, claim, entities)

        # Step 6: Select top 21 documents after reranking
        top_docs = [doc for doc, score in reranked_docs[:self.target_docs]]

        return dspy.Prediction(retrieved_docs=top_docs)
