import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractClaimEntities(dspy.Signature):
    """Extract 2-3 key entities or topics from the claim that are most important for fact verification."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="A list of 2-3 key entities, topics, or concepts from the claim that require verification")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused retrieval query for a specific entity in the context of the original claim."""

    claim: str = dspy.InputField()
    entity: str = dspy.InputField(desc="A specific entity or topic to focus on")
    query: str = dspy.OutputField(desc="A focused search query for this entity in the context of the claim")


class DocumentRelevanceScorer(dspy.Signature):
    """Score a document's relevance to a claim on a scale of 0-10, where 10 is highly relevant and 0 is completely irrelevant."""

    claim: str = dspy.InputField()
    document: str = dspy.InputField(desc="The document text to evaluate")
    score: int = dspy.OutputField(desc="Relevance score from 0 (irrelevant) to 10 (highly relevant)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize entity-focused multi-query strategy modules
        self.entity_extractor = dspy.Predict(ExtractClaimEntities)
        self.query_generator = dspy.Predict(EntityQueryGenerator)
        self.relevance_scorer = dspy.Predict(DocumentRelevanceScorer)
        self.retrieve_k25 = dspy.Retrieve(k=25)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract 2-3 key entities from the claim
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities

            # Ensure we have at most 3 entities
            if len(entities) > 3:
                entities = entities[:3]

            # Step 2: Generate one focused query per entity and retrieve documents
            all_docs = []
            for entity in entities:
                # Generate entity-focused query
                query_result = self.query_generator(claim=claim, entity=entity)
                query = query_result.query

                # Retrieve k=25 documents for this query
                docs = self.retrieve_k25(query).passages
                all_docs.extend(docs)

            # Step 3: Remove duplicate documents (by content)
            unique_docs = []
            seen_docs = set()
            for doc in all_docs:
                if doc not in seen_docs:
                    unique_docs.append(doc)
                    seen_docs.add(doc)

            # Step 4: Score each unique document for relevance to the original claim
            scored_docs = []
            for doc in unique_docs:
                try:
                    score_result = self.relevance_scorer(claim=claim, document=doc)
                    score = score_result.score
                    # Ensure score is an integer between 0 and 10
                    if isinstance(score, str):
                        score = int(score)
                    score = max(0, min(10, score))
                    scored_docs.append((doc, score))
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default mid-range score
                    scored_docs.append((doc, 5))

            # Step 5: Sort by score (descending) and return top 21 documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_21_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
