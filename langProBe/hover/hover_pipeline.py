import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityIdentificationSignature(dspy.Signature):
    """Extract 2-4 distinct named entities from the claim (people, places, organizations, works)."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="A list of 2-4 distinct named entities (people, places, organizations, or works) mentioned in the claim")


class DocumentRelevanceScorerSignature(dspy.Signature):
    """Score a document based on how well it supports verifying the claim."""

    claim: str = dspy.InputField()
    document: str = dspy.InputField()
    score: int = dspy.OutputField(desc="Relevance score from 0-10, where 10 means highly relevant for verifying the claim")


class DocumentRelevanceScorer(dspy.Module):
    """Scores documents based on relevance to the claim."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(DocumentRelevanceScorerSignature)

    def forward(self, claim, document):
        result = self.scorer(claim=claim, document=document)
        try:
            score = int(result.score)
            # Clamp score to 0-10 range
            score = max(0, min(10, score))
        except (ValueError, AttributeError):
            # Default to 5 if parsing fails
            score = 5
        return score


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.entity_identifier = dspy.ChainOfThought(EntityIdentificationSignature)
        self.relevance_scorer = DocumentRelevanceScorer()
        self.retrieve_k = dspy.Retrieve(k=10)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Identify entities from the claim
            entity_result = self.entity_identifier(claim=claim)
            entities = entity_result.entities

            # Ensure we have 2-4 entities (handle edge cases)
            if not entities or len(entities) == 0:
                # Fallback: use claim itself if no entities extracted
                entities = [claim]
            elif len(entities) > 4:
                # Limit to 4 entities to respect 3-search constraint
                entities = entities[:4]

            # Step 2: Perform parallel entity-specific retrieval
            # Collect all documents with their source entity
            all_docs_with_entity = []
            for entity in entities[:3]:  # Limit to 3 entities max (respects 3-search constraint)
                try:
                    docs = self.retrieve_k(entity).passages
                    for doc in docs:
                        all_docs_with_entity.append((doc, entity))
                except Exception:
                    # Handle retrieval failures gracefully
                    continue

            # Step 3: Score all retrieved documents
            scored_docs = []
            for doc, entity in all_docs_with_entity:
                try:
                    score = self.relevance_scorer(claim=claim, document=doc)
                    scored_docs.append((doc, entity, score))
                except Exception:
                    # If scoring fails, assign default score of 5
                    scored_docs.append((doc, entity, 5))

            # Step 4: Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x[2], reverse=True)

            # Step 5: Select top 21 documents with diversity constraint (max 8 per entity)
            selected_docs = []
            entity_counts = {}

            for doc, entity, score in scored_docs:
                if len(selected_docs) >= 21:
                    break

                # Check if we haven't exceeded the per-entity limit
                entity_count = entity_counts.get(entity, 0)
                if entity_count < 8:
                    selected_docs.append(doc)
                    entity_counts[entity] = entity_count + 1

            # Ensure we have exactly 21 documents if possible
            # If we have fewer, pad with remaining high-scoring docs
            if len(selected_docs) < 21:
                for doc, entity, score in scored_docs:
                    if doc not in selected_docs:
                        selected_docs.append(doc)
                        if len(selected_docs) >= 21:
                            break

            return dspy.Prediction(retrieved_docs=selected_docs[:21])
