import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from typing import List, Tuple
from collections import defaultdict


class EntityExtractor(dspy.Signature):
    """Extract ALL key entities from the claim that are critical for verification.

    Identify named entities (people, places, organizations), concepts, dates, and other
    important nouns that need to be verified through retrieval. Be comprehensive - include
    all entities that play a role in the claim's verification.
    """

    claim = dspy.InputField(desc="The claim that needs to be verified through multi-hop reasoning")

    reasoning = dspy.OutputField(desc="Explain what types of entities are present in the claim and why each is important for verification")
    entities = dspy.OutputField(desc="A comma-separated list of ALL key entities extracted from the claim (e.g., 'Entity1, Entity2, Entity3')")


class ChainOfThoughtQueryPlanner(dspy.Signature):
    """Analyze the claim and retrieved context to strategically plan the next retrieval query.

    Decompose what entities, relationships, and facts are needed to verify the claim.
    Analyze what information has already been found versus what's still missing.
    Generate a targeted query to find the specific missing information needed for the next hop.
    """

    claim = dspy.InputField(desc="The claim that needs to be verified through multi-hop reasoning")
    retrieved_context = dspy.InputField(desc="The context retrieved so far from previous hops (may be empty for first hop)")

    reasoning = dspy.OutputField(desc="Explain the multi-hop reasoning chain needed: what entities/relationships are mentioned in the claim and how they connect")
    missing_information = dspy.OutputField(desc="Identify specific gaps: what key information was found in retrieved_context vs. what's still needed to verify the claim")
    next_query = dspy.OutputField(desc="A focused search query to find the specific missing information identified above")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.entity_k = 7  # k=7 for entity-based retrieval
        self.max_retrieval_calls = 3  # Stay within 3-query limit
        self.target_docs = 21  # Target number of final documents

        # Stage 1: Entity extraction
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)

        # Stage 2: Entity-based retrieval
        self.retrieve_entity = dspy.Retrieve(k=self.entity_k)

    def _batch_entities(self, entities: List[str], max_batches: int) -> List[List[str]]:
        """Batch entities into max_batches groups for retrieval."""
        if len(entities) <= max_batches:
            return [[entity] for entity in entities]

        # Distribute entities as evenly as possible across batches
        batch_size = len(entities) // max_batches
        remainder = len(entities) % max_batches

        batches = []
        start_idx = 0
        for i in range(max_batches):
            # Add one extra entity to first 'remainder' batches
            current_batch_size = batch_size + (1 if i < remainder else 0)
            batches.append(entities[start_idx:start_idx + current_batch_size])
            start_idx += current_batch_size

        return batches

    def _compute_entity_overlap_score(self, doc: str, entities: List[str]) -> int:
        """Count how many entities are mentioned in the document."""
        doc_lower = doc.lower()
        return sum(1 for entity in entities if entity.lower() in doc_lower)

    def _deduplicate_and_rank_documents(
        self,
        doc_tuples: List[Tuple[str, float]],
        entities: List[str]
    ) -> List[str]:
        """
        Deduplicate and select top documents based on:
        1. Entity overlap score (documents mentioning multiple entities)
        2. Retrieval score from ColBERT

        Returns top self.target_docs documents.
        """
        # Deduplicate by doc content
        seen = set()
        unique_docs = []
        for doc, score in doc_tuples:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append((doc, score))

        # Compute combined score: entity_overlap (weighted heavily) + retrieval_score
        scored_docs = []
        for doc, retrieval_score in unique_docs:
            entity_overlap = self._compute_entity_overlap_score(doc, entities)
            # Prioritize entity overlap with high weight, then retrieval score
            combined_score = (entity_overlap * 10.0) + retrieval_score
            scored_docs.append((doc, combined_score, entity_overlap, retrieval_score))

        # Sort by combined score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top target_docs
        return [doc for doc, _, _, _ in scored_docs[:self.target_docs]]

    def forward(self, claim):
        # STAGE 1: Extract all key entities from the claim
        extraction_result = self.entity_extractor(claim=claim)

        # Parse entities from comma-separated string
        entities_str = extraction_result.entities.strip()
        entities = [e.strip() for e in entities_str.split(',') if e.strip()]

        # STAGE 2: Perform entity-based retrieval
        # Batch entities to stay within max_retrieval_calls limit
        entity_batches = self._batch_entities(entities, self.max_retrieval_calls)

        all_doc_tuples = []  # List of (document, score) tuples

        for batch in entity_batches:
            # Create a query from the batch of entities
            if len(batch) == 1:
                query = batch[0]
            else:
                # Combine multiple entities into a single query
                query = " ".join(batch)

            # Retrieve documents for this entity batch
            retrieval_result = self.retrieve_entity(query)

            # Store documents with their retrieval scores
            # Note: dspy.Retrieve may not always provide scores, so we use index as fallback
            for idx, doc in enumerate(retrieval_result.passages):
                # Higher index = lower score (reverse ranking)
                pseudo_score = float(len(retrieval_result.passages) - idx)
                all_doc_tuples.append((doc, pseudo_score))

        # STAGE 3: Deduplicate and rank documents
        # Prioritize documents mentioning multiple entities and with higher retrieval scores
        final_docs = self._deduplicate_and_rank_documents(all_doc_tuples, entities)

        return dspy.Prediction(
            retrieved_docs=final_docs,
            extracted_entities=entities,
            entity_reasoning=extraction_result.reasoning
        )


