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




class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    APPROACH
    - Entity-extraction-first: Extract all entities, relationships, and facts from the claim
    - Create targeted queries for entity clusters
    - Use k=50 for primary entities, k=40 for secondary/bridging, k=30 for relationship verification
    - Apply deterministic reranking based on lexical overlap and retrieval position to select final 21 unique documents
    '''

    def __init__(self):
        super().__init__()

        # Entity extraction module
        self.entity_extractor = dspy.ChainOfThought(ClaimEntityExtractor)

        # Query generation for each hop
        self.query_generator = dspy.ChainOfThought(EntityBasedQueryGenerator)

        # Retrieval modules with different k values for each hop
        self.retrieve_hop1 = dspy.Retrieve(k=50)  # Primary entities
        self.retrieve_hop2 = dspy.Retrieve(k=40)  # Secondary/bridging entities
        self.retrieve_hop3 = dspy.Retrieve(k=30)  # Relationship verification

    def _compute_lexical_overlap(self, claim, document):
        """Compute lexical overlap score between claim and document text.

        Args:
            claim: The claim text
            document: The document text

        Returns:
            Float score representing the lexical overlap
        """
        # Normalize and tokenize
        claim_words = set(claim.lower().split())
        doc_words = set(document.lower().split())

        # Remove common stop words that don't carry much meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                     'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}

        claim_words = claim_words - stop_words
        doc_words = doc_words - stop_words

        # Compute overlap
        if not claim_words:
            return 0.0

        overlap = len(claim_words & doc_words)
        # Normalize by claim length (Jaccard-like similarity)
        score = overlap / len(claim_words)

        return score

    def _extract_document_title(self, document):
        """Extract document title from the document string.

        Documents are formatted as "Title | Content", so we extract the title part.

        Args:
            document: The full document string

        Returns:
            The document title (part before " | ")
        """
        if ' | ' in document:
            return document.split(' | ')[0]
        return document[:100]  # Fallback: use first 100 chars as title

    def _deduplicate_documents(self, documents):
        """Deduplicate documents by title.

        Args:
            documents: List of (doc, position) tuples

        Returns:
            List of deduplicated (doc, position) tuples, keeping first occurrence
        """
        seen_titles = set()
        deduplicated = []

        for doc, pos in documents:
            title = self._extract_document_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append((doc, pos))

        return deduplicated

    def _score_document(self, claim, document, position):
        """Compute deterministic relevance score for a document.

        Args:
            claim: The claim text
            document: The document text
            position: Position in original retrieval (0-indexed, lower is better)

        Returns:
            Combined score for ranking
        """
        # Lexical overlap score (0.0 to 1.0+)
        lexical_score = self._compute_lexical_overlap(claim, document)

        # Position score: earlier positions get higher scores
        # Normalize position to 0-1 range with exponential decay
        # Position 0 gets score ~1.0, position 100 gets score ~0.0
        position_score = 1.0 / (1.0 + position * 0.1)

        # Combine scores: weight lexical overlap more heavily (70%) vs position (30%)
        combined_score = 0.7 * lexical_score + 0.3 * position_score

        return combined_score

    def forward(self, claim):
        # STEP 1: Extract entities, relationships, and key facts from the claim
        extraction = self.entity_extractor(claim=claim)
        primary_entities = extraction.primary_entities
        secondary_entities = extraction.secondary_entities
        relationships = extraction.relationships
        key_facts = extraction.key_facts

        # HOP 1: Retrieve documents for primary entities (k=50)
        hop1_query = self.query_generator(
            claim=claim,
            entity_cluster=primary_entities,
            relationships=relationships,
            previous_findings=""
        ).query
        hop1_docs = self.retrieve_hop1(hop1_query).passages

        # Summarize hop 1 findings
        hop1_summary = f"Found documents about primary entities: {primary_entities[:200]}"

        # HOP 2: Retrieve documents for secondary/bridging entities (k=40)
        hop2_query = self.query_generator(
            claim=claim,
            entity_cluster=secondary_entities,
            relationships=relationships,
            previous_findings=hop1_summary
        ).query
        hop2_docs = self.retrieve_hop2(hop2_query).passages

        # Summarize hop 2 findings
        hop2_summary = f"Found documents about secondary entities and connections"

        # HOP 3: Retrieve documents for relationship verification (k=30)
        # Focus on verifying relationships and key facts
        relationship_cluster = f"Relationships: {relationships}; Key facts: {key_facts}"
        hop3_query = self.query_generator(
            claim=claim,
            entity_cluster=relationship_cluster,
            relationships=relationships,
            previous_findings=f"{hop1_summary}; {hop2_summary}"
        ).query
        hop3_docs = self.retrieve_hop3(hop3_query).passages

        # STEP 2: Combine all retrieved documents with position tracking (total: 50 + 40 + 30 = 120)
        # Track position in original retrieval for scoring
        docs_with_positions = []
        for i, doc in enumerate(hop1_docs):
            docs_with_positions.append((doc, i))
        for i, doc in enumerate(hop2_docs):
            docs_with_positions.append((doc, i + 50))
        for i, doc in enumerate(hop3_docs):
            docs_with_positions.append((doc, i + 90))

        # STEP 3: Deduplicate documents by title
        unique_docs = self._deduplicate_documents(docs_with_positions)

        # STEP 4: Score documents using deterministic scoring
        scored_docs = []
        for doc, position in unique_docs:
            score = self._score_document(claim, doc, position)
            scored_docs.append((doc, score))

        # STEP 5: Sort by score (descending) and select top 21
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=final_docs)
