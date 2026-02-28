import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityIdentification(dspy.Signature):
    """Extract all proper nouns (people, bands, works, organizations, places) from the claim that should be looked up in Wikipedia."""

    claim: str = dspy.InputField(desc="A claim that needs to be fact-checked")
    entities: list[str] = dspy.OutputField(desc="List of proper nouns to search for, formatted as Wikipedia article titles (e.g., 'Roger Waters', 'Tom Johnston musician', 'Martin Flavin')")


class DocumentRelevanceScorer(dspy.Signature):
    """Score a document's relevance to the identified entities and claim relationships."""

    claim: str = dspy.InputField(desc="The original claim to fact-check")
    entities: str = dspy.InputField(desc="Comma-separated list of identified entities")
    document: str = dspy.InputField(desc="The document content to score")
    relevance_score: float = dspy.OutputField(desc="Relevance score from 0.0 to 10.0, where 10.0 means the document contains biographical/definitional information about key entities or direct relationships between entities mentioned in the claim")


class EntityQueryGenerator(dspy.Signature):
    """Generate Wikipedia article title queries for the identified entities."""

    claim: str = dspy.InputField(desc="The claim to fact-check")
    entities: str = dspy.InputField(desc="Comma-separated list of entities to query")
    queries: list[str] = dspy.OutputField(desc="List of entity-specific queries formatted as direct Wikipedia article titles")


class BridgingQueryGenerator(dspy.Signature):
    """Generate a bridging query that connects the identified entities based on the claim."""

    claim: str = dspy.InputField(desc="The claim to fact-check")
    entities: str = dspy.InputField(desc="Comma-separated list of identified entities")
    hop1_summary: str = dspy.InputField(desc="Summary of documents from first hop")
    query: str = dspy.OutputField(desc="A bridging query that connects the entities")


class VerificationQueryGenerator(dspy.Signature):
    """Generate a verification query to confirm or refute the claim."""

    claim: str = dspy.InputField(desc="The claim to fact-check")
    entities: str = dspy.InputField(desc="Comma-separated list of identified entities")
    hop1_summary: str = dspy.InputField(desc="Summary of documents from first hop")
    hop2_summary: str = dspy.InputField(desc="Summary of documents from second hop")
    query: str = dspy.OutputField(desc="A verification query to check the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Entity-based retrieval modules
        self.entity_identifier = dspy.Predict(EntityIdentification)
        self.entity_query_gen = dspy.Predict(EntityQueryGenerator)
        self.bridging_query_gen = dspy.ChainOfThought(BridgingQueryGenerator)
        self.verification_query_gen = dspy.ChainOfThought(VerificationQueryGenerator)

        # Retrieval modules with different k values
        self.retrieve_15 = dspy.Retrieve(k=15)
        self.retrieve_5 = dspy.Retrieve(k=5)

        # Summarization modules
        self.summarize_hop1 = dspy.ChainOfThought("claim,entities,passages->summary")
        self.summarize_hop2 = dspy.ChainOfThought("claim,entities,context,passages->summary")

        # Document scoring module
        self.doc_scorer = dspy.Predict(DocumentRelevanceScorer)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Identify entities from the claim
            entity_result = self.entity_identifier(claim=claim)
            entities = entity_result.entities
            entities_str = ", ".join(entities) if isinstance(entities, list) else str(entities)

            # Step 2: HOP 1 - Entity-specific queries (k=15)
            # Generate entity-specific queries formatted as Wikipedia article titles
            entity_queries_result = self.entity_query_gen(claim=claim, entities=entities_str)
            entity_queries = entity_queries_result.queries

            # Retrieve using the first entity query or claim if no entities found
            if isinstance(entity_queries, list) and len(entity_queries) > 0:
                hop1_query = entity_queries[0]
            else:
                hop1_query = claim

            hop1_docs = self.retrieve_15(hop1_query).passages

            # Summarize hop 1 documents
            hop1_summary = self.summarize_hop1(
                claim=claim,
                entities=entities_str,
                passages=hop1_docs
            ).summary

            # Step 3: HOP 2 - Bridging query (k=15)
            hop2_query = self.bridging_query_gen(
                claim=claim,
                entities=entities_str,
                hop1_summary=hop1_summary
            ).query
            hop2_docs = self.retrieve_15(hop2_query).passages

            # Summarize hop 2 documents
            hop2_summary = self.summarize_hop2(
                claim=claim,
                entities=entities_str,
                context=hop1_summary,
                passages=hop2_docs
            ).summary

            # Step 4: HOP 3 - Verification query (k=5)
            hop3_query = self.verification_query_gen(
                claim=claim,
                entities=entities_str,
                hop1_summary=hop1_summary,
                hop2_summary=hop2_summary
            ).query
            hop3_docs = self.retrieve_5(hop3_query).passages

            # Step 5: Score and rerank all 35 documents (15+15+5)
            all_docs = hop1_docs + hop2_docs + hop3_docs
            scored_docs = []

            for doc in all_docs:
                try:
                    score_result = self.doc_scorer(
                        claim=claim,
                        entities=entities_str,
                        document=doc[:500]  # Limit doc length for scoring
                    )
                    score = float(score_result.relevance_score)
                except (ValueError, AttributeError):
                    # If scoring fails, assign a default middle score
                    score = 5.0

                scored_docs.append((score, doc))

            # Sort by score (descending) and take top 21
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_21_docs = [doc for score, doc in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
