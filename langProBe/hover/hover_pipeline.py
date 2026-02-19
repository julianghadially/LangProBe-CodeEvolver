import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractor(dspy.Signature):
    """Extract 3-5 key entities, concepts, titles, or names from the claim that would be helpful for document retrieval."""

    claim: str = dspy.InputField(desc="the claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="list of 3-5 key entities, concepts, titles, or names from the claim")


class QueryFromEntity(dspy.Signature):
    """Generate a specific and targeted search query for a given entity in the context of the claim."""

    claim: str = dspy.InputField(desc="the original claim to verify")
    entity: str = dspy.InputField(desc="the entity or concept to focus the query on")
    query: str = dspy.OutputField(desc="a specific search query targeting the entity in context of the claim")


class DocumentReranker(dspy.Signature):
    """Score a document's relevance to a claim on a scale from 1 to 10, where 10 is highly relevant and 1 is not relevant."""

    claim: str = dspy.InputField(desc="the claim to verify")
    document: str = dspy.InputField(desc="the document to score for relevance")
    score: int = dspy.OutputField(desc="relevance score from 1 (not relevant) to 10 (highly relevant)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize sub-modules for entity-based retrieval
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.query_generator = dspy.ChainOfThought(QueryFromEntity)
        self.reranker = dspy.Predict(DocumentReranker)

        # Set k=5 for per-entity retrieval
        self.retrieve_k = dspy.Retrieve(k=5)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract 3-5 key entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities

            # Ensure we have 3-5 entities
            if len(entities) < 3:
                # If fewer than 3, pad with the claim itself
                entities = entities + [claim] * (3 - len(entities))
            elif len(entities) > 5:
                # If more than 5, take only first 5
                entities = entities[:5]

            # Step 2: Generate targeted query for each entity and retrieve documents
            all_docs = []
            for entity in entities:
                # Generate query for this entity
                query_result = self.query_generator(claim=claim, entity=entity)
                entity_query = query_result.query

                # Retrieve k=5 documents for this entity query
                entity_docs = self.retrieve_k(entity_query).passages
                all_docs.extend(entity_docs)

            # Step 3: Deduplicate the combined results
            # Documents are in format "title | content", so we deduplicate by full string
            unique_docs = deduplicate(all_docs)

            # Step 4: Rerank all unique documents
            scored_docs = []
            for doc in unique_docs:
                # Score each document
                score_result = self.reranker(claim=claim, document=doc)
                score = score_result.score
                scored_docs.append((score, doc))

            # Sort by score (descending) and take top 21
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_docs)
