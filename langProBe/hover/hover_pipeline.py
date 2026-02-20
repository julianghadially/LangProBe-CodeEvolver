import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractorSignature(dspy.Signature):
    """Extract 2-4 key entities or topics from the claim that require verification.
    Focus on specific names (people, organizations), titles, events, or verifiable facts."""

    claim: str = dspy.InputField(desc="The claim to verify")
    entities: list[str] = dspy.OutputField(
        desc="A list of 2-4 key entities/topics (e.g., person names, organization names, specific titles, events)"
    )


class FocusedQueryGeneratorSignature(dspy.Signature):
    """Generate a highly specific search query focused on a single entity from the claim."""

    claim: str = dspy.InputField(desc="The original claim to verify")
    entity: str = dspy.InputField(desc="The specific entity/topic to search for")
    query: str = dspy.OutputField(desc="A focused search query for this entity")


class DocumentRelevanceSignature(dspy.Signature):
    """Score how relevant a document is to verifying the original claim on a 0-10 scale."""

    claim: str = dspy.InputField(desc="The claim to verify")
    document: str = dspy.InputField(desc="The document to score")
    score: int = dspy.OutputField(desc="Relevance score from 0 (not relevant) to 10 (highly relevant)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Entity-focused retrieval modules
        self.entity_extractor = dspy.ChainOfThought(EntityExtractorSignature)
        self.query_generator = dspy.ChainOfThought(FocusedQueryGeneratorSignature)
        self.retrieve_k100 = dspy.Retrieve(k=100)
        self.document_scorer = dspy.Predict(DocumentRelevanceSignature)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract 2-4 key entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities

            # Limit to maximum 3 entities for query generation (respecting constraints)
            entities = entities[:3]

            # Step 2 & 3: For each entity, generate a focused query and retrieve k=100 documents
            all_docs = []

            for entity in entities:
                # Generate focused query for this entity
                query_result = self.query_generator(claim=claim, entity=entity)
                query = query_result.query

                # Retrieve k=100 documents for this query
                retrieved_docs = self.retrieve_k100(query).passages

                # Step 4: Rerank documents using LLM scoring
                scored_docs = []
                for doc in retrieved_docs:
                    try:
                        score_result = self.document_scorer(claim=claim, document=doc)
                        score = int(score_result.score) if hasattr(score_result, 'score') else 0
                        # Clamp score to 0-10 range
                        score = max(0, min(10, score))
                        scored_docs.append((score, doc))
                    except:
                        # If scoring fails, assign a default low score
                        scored_docs.append((0, doc))

                # Sort by score (descending) and select top 7
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_7_docs = [doc for score, doc in scored_docs[:7]]
                all_docs.extend(top_7_docs)

            # Step 5: Deduplicate and return exactly 21 documents
            unique_docs = deduplicate(all_docs)

            # Ensure exactly 21 documents (pad if needed, truncate if over)
            if len(unique_docs) < 21:
                # If we have fewer than 21, keep what we have
                final_docs = unique_docs
            else:
                # If we have more, take the first 21
                final_docs = unique_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
