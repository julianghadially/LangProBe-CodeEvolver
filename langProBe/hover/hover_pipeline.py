import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityQueryGenerator(dspy.Signature):
    """Generate a query that focuses on extracting and querying key entities from the claim."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query focused on key entities mentioned in the claim")


class RelationQueryGenerator(dspy.Signature):
    """Generate a query that focuses on relationships between entities in the claim."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query focused on relationships between entities in the claim")


class TemporalQueryGenerator(dspy.Signature):
    """Generate a query that emphasizes temporal or comparative aspects of the claim."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query focused on temporal or comparative aspects of the claim")


class DocumentConfidenceScorer(dspy.Signature):
    """Score the relevance of a document to a claim with a confidence score."""
    claim: str = dspy.InputField()
    document: str = dspy.InputField()
    confidence: float = dspy.OutputField(desc="confidence score from 0.0 to 1.0 indicating relevance of the document to the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize query generators
        self.entity_query_gen = dspy.ChainOfThought(EntityQueryGenerator)
        self.relation_query_gen = dspy.ChainOfThought(RelationQueryGenerator)
        self.temporal_query_gen = dspy.ChainOfThought(TemporalQueryGenerator)

        # Initialize retriever with k=25
        self.retrieve_k25 = dspy.Retrieve(k=25)

        # Initialize confidence scorer
        self.scorer = dspy.Predict(DocumentConfidenceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate three different query types in parallel
            entity_query = self.entity_query_gen(claim=claim).query
            relation_query = self.relation_query_gen(claim=claim).query
            temporal_query = self.temporal_query_gen(claim=claim).query

            # Retrieve k=25 documents per query (75 total)
            entity_docs = self.retrieve_k25(entity_query).passages
            relation_docs = self.retrieve_k25(relation_query).passages
            temporal_docs = self.retrieve_k25(temporal_query).passages

            # Combine all documents
            all_docs = entity_docs + relation_docs + temporal_docs

            # Score all documents with batch processing
            doc_scores = []
            for doc in all_docs:
                try:
                    result = self.scorer(claim=claim, document=doc)
                    score = float(result.confidence)
                    # Ensure score is in valid range
                    score = max(0.0, min(1.0, score))
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default low score
                    score = 0.0
                doc_scores.append((doc, score))

            # Deduplicate documents by title (keeping highest score)
            title_to_best = {}
            for doc, score in doc_scores:
                # Extract title from document (format: "title | content")
                title = doc.split(" | ")[0] if " | " in doc else doc
                if title not in title_to_best or score > title_to_best[title][1]:
                    title_to_best[title] = (doc, score)

            # Sort by confidence score descending
            sorted_docs = sorted(title_to_best.values(), key=lambda x: x[1], reverse=True)

            # Return top 21 unique documents
            top_21_docs = [doc for doc, score in sorted_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
