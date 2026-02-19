import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimEntityParser(dspy.Signature):
    """Extract 2-4 specific named entities, phrases, or titles from the claim that are most likely to appear in supporting documents. Focus on proper nouns, titles, names of people, places, works, events, or organizations."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="A list of 2-4 specific named entities, phrases, or titles extracted from the claim")


class EntityCoverageReranker(dspy.Signature):
    """Score each document (0-10) based on: (1) exact entity name matches, (2) definitional content about entities (e.g., 'X is a...', 'X was born...'), and (3) relationship information connecting entities. Higher scores for documents that directly define or describe the key entities."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.InputField(desc="List of extracted entities from the claim")
    documents: list[str] = dspy.InputField(desc="List of retrieved documents to score")
    scores: list[int] = dspy.OutputField(desc="List of scores (0-10) for each document, where higher scores indicate better entity coverage and definitional content")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using two-stage targeted entity retrieval with constraint-aware reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    APPROACH
    - Stage 1: Extract 2-4 entities from the claim and retrieve k=60 documents using a combined query
    - Stage 2: Rerank documents based on entity coverage, definitional content, and relationships, then select top 21
    '''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.entity_parser = dspy.Predict(ClaimEntityParser)
        self.reranker = dspy.Predict(EntityCoverageReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Entity Extraction & Direct Retrieval
            # Extract 2-4 specific entities from the claim
            parsed = self.entity_parser(claim=claim)
            entities = parsed.entities if hasattr(parsed, 'entities') else []

            # Construct combined query string
            if entities and isinstance(entities, list) and len(entities) > 0:
                query = " OR ".join(entities)
            else:
                # Fallback to using the claim if entity extraction fails
                query = claim

            # Retrieve k=60 documents in one search
            retrieved = dspy.Retrieve(k=60)(query)
            documents = retrieved.passages

            # Stage 2: Constraint-Aware Reranking
            # Score all 60 documents based on entity coverage
            reranked = self.reranker(
                claim=claim,
                entities=entities if entities else [claim],
                documents=documents
            )

            # Get scores and sort documents
            scores = reranked.scores if hasattr(reranked, 'scores') else []

            # Pair documents with scores, handling cases where scores may not match document count
            if scores and isinstance(scores, list) and len(scores) == len(documents):
                doc_score_pairs = list(zip(documents, scores))
                # Sort by score in descending order
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                # Take top 21 documents
                top_docs = [doc for doc, score in doc_score_pairs[:21]]
            else:
                # Fallback: if reranking fails, just take top 21 from retrieval
                top_docs = documents[:21]

            return dspy.Prediction(retrieved_docs=top_docs)
