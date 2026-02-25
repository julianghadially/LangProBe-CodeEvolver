import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractCoreEntities(dspy.Signature):
    """Extract key named entities and important concepts from a claim for entity-focused retrieval.
    Focus on people, places, organizations, dates, and specific technical terms."""

    claim: str = dspy.InputField()
    entities: str = dspy.OutputField(desc="comma-separated list of key entities and concepts from the claim")


class ExtractRelationships(dspy.Signature):
    """Extract relationships, connections, and interactions described in a claim for relationship-focused retrieval.
    Focus on verbs, actions, causal connections, and how entities relate to each other."""

    claim: str = dspy.InputField()
    relationships: str = dspy.OutputField(desc="description of key relationships and connections in the claim")


class ScoreDocumentRelevance(dspy.Signature):
    """Score how relevant a document is to verifying a given claim.
    Consider whether the document contains information that would help verify or refute the claim."""

    claim: str = dspy.InputField()
    document: str = dspy.InputField(desc="document title and passage")
    score: int = dspy.OutputField(desc="relevance score from 1 (not relevant) to 10 (highly relevant)")
    justification: str = dspy.OutputField(desc="brief explanation of the score")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 12  # Each pathway retrieves 12 documents

        # Initialize retriever
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Initialize the three specialist pathways
        self.entity_extractor = dspy.Predict(ExtractCoreEntities)
        self.relationship_extractor = dspy.Predict(ExtractRelationships)

        # Initialize document scorer
        self.document_scorer = dspy.Predict(ScoreDocumentRelevance)

    def forward(self, claim):
        # PATHWAY 1: Entity-focused retrieval
        # Extract entities and retrieve documents focused on those entities
        entity_result = self.entity_extractor(claim=claim)
        entity_query = entity_result.entities
        entity_docs = self.retrieve_k(entity_query).passages

        # PATHWAY 2: Relationship-focused retrieval
        # Extract relationships and retrieve documents focused on those relationships
        relationship_result = self.relationship_extractor(claim=claim)
        relationship_query = relationship_result.relationships
        relationship_docs = self.retrieve_k(relationship_query).passages

        # PATHWAY 3: Direct retrieval
        # Retrieve documents directly from the claim without transformation
        direct_docs = self.retrieve_k(claim).passages

        # Combine all documents (~36 total)
        all_docs = entity_docs + relationship_docs + direct_docs

        # Score each document for relevance
        scored_docs = []
        for doc in all_docs:
            try:
                score_result = self.document_scorer(claim=claim, document=doc)
                # Parse score to ensure it's an integer
                try:
                    score = int(score_result.score)
                except (ValueError, TypeError):
                    score = 5  # Default middle score if parsing fails

                # Extract document title (format: "title | passage")
                title = doc.split(" | ")[0] if " | " in doc else doc

                scored_docs.append({
                    'doc': doc,
                    'title': title,
                    'score': score
                })
            except Exception:
                # If scoring fails for any reason, assign default score
                title = doc.split(" | ")[0] if " | " in doc else doc
                scored_docs.append({
                    'doc': doc,
                    'title': title,
                    'score': 5
                })

        # Deduplicate by title, keeping the highest score for each unique title
        title_to_best_doc = {}
        for item in scored_docs:
            title = item['title']
            if title not in title_to_best_doc or item['score'] > title_to_best_doc[title]['score']:
                title_to_best_doc[title] = item

        # Get unique documents sorted by score (descending)
        unique_docs = list(title_to_best_doc.values())
        unique_docs.sort(key=lambda x: x['score'], reverse=True)

        # Select top 21 documents
        top_21_docs = [item['doc'] for item in unique_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
