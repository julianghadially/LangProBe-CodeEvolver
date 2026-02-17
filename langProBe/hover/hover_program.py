import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimEntityExtractor(dspy.Signature):
    """Extract all named entities, relationships, and key facts from a claim.
    Identify:
    1. Named entities (people, places, organizations, dates, events)
    2. Relationships between entities
    3. Key facts that need verification

    Group related entities into clusters (primary, secondary, bridging entities)."""

    claim = dspy.InputField(desc="The claim to analyze for entity extraction")

    primary_entities = dspy.OutputField(
        desc="Core entities central to the claim (people, places, organizations most directly mentioned)"
    )
    secondary_entities = dspy.OutputField(
        desc="Supporting entities that provide context or connect to primary entities"
    )
    relationships = dspy.OutputField(
        desc="Key relationships between entities that need to be verified"
    )
    key_facts = dspy.OutputField(
        desc="Specific facts, dates, or events that require verification"
    )


class EntityBasedQueryGenerator(dspy.Signature):
    """Generate a focused search query to retrieve documents about specific entities and their relationships.
    The query should be optimized to find relevant documents that cover the specified entities."""

    claim = dspy.InputField(desc="The original claim being verified")
    entity_cluster = dspy.InputField(desc="The group of entities to focus this query on")
    relationships = dspy.InputField(desc="Relationships between entities to explore")
    previous_findings = dspy.InputField(
        desc="Summary of what has been found in previous searches (empty for first hop)"
    )

    query = dspy.OutputField(
        desc="A targeted search query to find documents about these entities and relationships"
    )


class DocumentRelevanceScorer(dspy.Signature):
    """Score how well a document covers the extracted entities and relationships.
    Consider:
    1. How many of the target entities are mentioned
    2. Whether key relationships are discussed
    3. Whether key facts are verified or contradicted

    Return a relevance score from 0-10."""

    claim = dspy.InputField(desc="The original claim being verified")
    document = dspy.InputField(desc="The document to score")
    target_entities = dspy.InputField(desc="All entities extracted from the claim")
    target_relationships = dspy.InputField(desc="Relationships that need verification")

    relevance_score = dspy.OutputField(
        desc="Relevance score from 0-10, where 10 means the document covers many entities and relationships"
    )
    covered_entities = dspy.OutputField(
        desc="Which entities from the target list are covered in this document"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    APPROACH
    - Entity-extraction-first: Extract all entities, relationships, and facts from the claim
    - Create targeted queries for entity clusters
    - Use k=30 for primary entities, k=20 for secondary/bridging, k=15 for relationship verification
    - Apply relevance-based reranking to select final 21 documents with maximum entity coverage
    '''

    def __init__(self):
        super().__init__()

        # Entity extraction module
        self.entity_extractor = dspy.ChainOfThought(ClaimEntityExtractor)

        # Query generation for each hop
        self.query_generator = dspy.ChainOfThought(EntityBasedQueryGenerator)

        # Document relevance scoring for reranking
        self.doc_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)

        # Retrieval modules with different k values for each hop
        self.retrieve_hop1 = dspy.Retrieve(k=30)  # Primary entities
        self.retrieve_hop2 = dspy.Retrieve(k=20)  # Secondary/bridging entities
        self.retrieve_hop3 = dspy.Retrieve(k=15)  # Relationship verification

    def forward(self, claim):
        # STEP 1: Extract entities, relationships, and key facts from the claim
        extraction = self.entity_extractor(claim=claim)
        primary_entities = extraction.primary_entities
        secondary_entities = extraction.secondary_entities
        relationships = extraction.relationships
        key_facts = extraction.key_facts

        # Combine all entities for tracking coverage
        all_entities = f"Primary: {primary_entities}; Secondary: {secondary_entities}"

        # HOP 1: Retrieve documents for primary entities (k=30)
        hop1_query = self.query_generator(
            claim=claim,
            entity_cluster=primary_entities,
            relationships=relationships,
            previous_findings=""
        ).query
        hop1_docs = self.retrieve_hop1(hop1_query).passages

        # Summarize hop 1 findings
        hop1_summary = f"Found documents about primary entities: {primary_entities[:200]}"

        # HOP 2: Retrieve documents for secondary/bridging entities (k=20)
        hop2_query = self.query_generator(
            claim=claim,
            entity_cluster=secondary_entities,
            relationships=relationships,
            previous_findings=hop1_summary
        ).query
        hop2_docs = self.retrieve_hop2(hop2_query).passages

        # Summarize hop 2 findings
        hop2_summary = f"Found documents about secondary entities and connections"

        # HOP 3: Retrieve documents for relationship verification (k=15)
        # Focus on verifying relationships and key facts
        relationship_cluster = f"Relationships: {relationships}; Key facts: {key_facts}"
        hop3_query = self.query_generator(
            claim=claim,
            entity_cluster=relationship_cluster,
            relationships=relationships,
            previous_findings=f"{hop1_summary}; {hop2_summary}"
        ).query
        hop3_docs = self.retrieve_hop3(hop3_query).passages

        # STEP 2: Combine all retrieved documents (total: 30 + 20 + 15 = 65)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # STEP 3: Rerank documents based on entity coverage and relevance
        # Score each document and select top 21
        scored_docs = []

        for doc in all_docs:
            try:
                # Score the document based on entity coverage
                scoring = self.doc_scorer(
                    claim=claim,
                    document=doc[:500],  # Limit doc length for scoring
                    target_entities=all_entities,
                    target_relationships=relationships
                )
                score = float(scoring.relevance_score) if hasattr(scoring, 'relevance_score') else 5.0
                covered = scoring.covered_entities if hasattr(scoring, 'covered_entities') else ""
                scored_docs.append((doc, score, covered))
            except:
                # If scoring fails, assign a default moderate score
                scored_docs.append((doc, 5.0, ""))

        # Sort by relevance score (descending) and select top 21
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Select top 21 documents that maximize entity coverage
        # Use a greedy algorithm to ensure diverse entity coverage
        final_docs = []
        covered_entities_set = set()

        # First pass: select highest scoring docs
        for doc, score, covered in scored_docs:
            if len(final_docs) >= 21:
                break
            final_docs.append(doc)
            # Track which entities we've covered
            if covered:
                covered_entities_set.update(covered.split(','))

        return dspy.Prediction(retrieved_docs=final_docs)
