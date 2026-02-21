import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractor(dspy.Signature):
    """Extract up to 3 key named entities (people, organizations, locations, events) from the claim that are critical for fact-checking."""

    claim: str = dspy.InputField(desc="the claim to fact-check")
    entities: list[str] = dspy.OutputField(desc="list of up to 3 key named entities (people, organizations, locations, events) extracted from the claim")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused search query for a specific entity in the context of the claim."""

    claim: str = dspy.InputField(desc="the original claim to fact-check")
    entity: str = dspy.InputField(desc="the specific entity to create a query for")
    query: str = dspy.OutputField(desc="a focused search query to retrieve documents about this entity in relation to the claim")


class EntityCoverageReranker(dspy.Signature):
    """Score a document based on how many extracted entities it contains and its semantic relevance to the claim."""

    claim: str = dspy.InputField(desc="the original claim to fact-check")
    entities: list[str] = dspy.InputField(desc="list of key entities extracted from the claim")
    document_title: str = dspy.InputField(desc="the title of the document to score")
    document_text: str = dspy.InputField(desc="the text content of the document")
    score: float = dspy.OutputField(desc="relevance score from 0.0 to 1.0 based on entity coverage and semantic relevance to the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize entity-based retrieval components
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.query_generator = dspy.Predict(EntityQueryGenerator)
        self.reranker = dspy.Predict(EntityCoverageReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract up to 3 key entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities[:3]  # Ensure max 3 entities

            # If no entities extracted or list is empty, use direct claim query
            if not entities:
                entities = [claim]

            # Step 2: Generate one query per entity (max 3 queries)
            all_documents = []
            seen_titles = set()

            for entity in entities:
                # Generate entity-specific query
                query_result = self.query_generator(claim=claim, entity=entity)
                query = query_result.query

                # Retrieve k=70 documents for this query
                retrieved = self.rm(query, k=70)

                # Deduplicate by document title
                for doc in retrieved:
                    # Extract title from document (format: "title | text")
                    doc_parts = doc.split(" | ", 1)
                    doc_title = doc_parts[0] if doc_parts else doc

                    # Add only if we haven't seen this title
                    if doc_title not in seen_titles:
                        seen_titles.add(doc_title)
                        all_documents.append(doc)

            # Step 3: Rerank combined documents based on entity coverage + semantic relevance
            scored_docs = []

            for doc in all_documents:
                # Extract title and text from document
                doc_parts = doc.split(" | ", 1)
                doc_title = doc_parts[0] if doc_parts else doc
                doc_text = doc_parts[1] if len(doc_parts) > 1 else ""

                # Score the document
                score_result = self.reranker(
                    claim=claim,
                    entities=entities,
                    document_title=doc_title,
                    document_text=doc_text
                )

                # Parse score (handle potential string output)
                try:
                    score = float(score_result.score)
                except (ValueError, TypeError):
                    score = 0.5  # Default middle score if parsing fails

                scored_docs.append((score, doc))

            # Sort by score (descending) and take top 21
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_21_docs = [doc for score, doc in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
