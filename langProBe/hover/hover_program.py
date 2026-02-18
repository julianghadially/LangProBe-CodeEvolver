import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractClaimEntities(dspy.Signature):
    """Extract all named entities (people, organizations, places, titles) from the input claim.

    Identify and extract:
    - People: Names of individuals mentioned
    - Organizations: Companies, institutions, groups
    - Places: Countries, cities, locations
    - Titles: Books, movies, songs, articles, events

    Return entities as a structured list."""

    claim = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="List of named entities extracted from the claim")


class VerifyEntityCoverage(dspy.Signature):
    """Analyze which entities from the claim have zero or minimal coverage in the retrieved documents.

    Compare the list of entities against the retrieved documents and identify entities that:
    - Are not mentioned at all in any document
    - Are mentioned but lack sufficient context or information
    - Require additional information to verify the claim

    Return uncovered entities ranked by importance for claim verification."""

    claim = dspy.InputField(desc="The original claim being verified")
    entities = dspy.InputField(desc="List of entities extracted from the claim")
    documents = dspy.InputField(desc="Currently retrieved documents")
    uncovered_entities: list[str] = dspy.OutputField(desc="List of entities with zero or minimal coverage, ranked by importance")


class RankDocumentsByRelevance(dspy.Signature):
    """Score and rank documents based on entity coverage and claim alignment.

    Evaluate each document by:
    - Entity coverage: How many claim entities are mentioned
    - Claim alignment: How relevant the content is to verifying the claim
    - Information density: Quality and depth of information provided

    Return relevance scores for ranking."""

    claim = dspy.InputField(desc="The claim being verified")
    entities = dspy.InputField(desc="List of entities from the claim")
    documents = dspy.InputField(desc="All retrieved documents to rank")
    relevance_scores: list[float] = dspy.OutputField(desc="Relevance scores for each document (0.0 to 1.0)")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverEntityAwareMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Entity-aware gap analysis multi-hop retrieval system for claims.

    This system implements a sophisticated retrieval pipeline that:
    1. Extracts entities from the claim
    2. Retrieves initial documents with full claim query
    3. Identifies entities with zero or minimal coverage (gap analysis)
    4. Performs targeted retrieval for uncovered entities
    5. Reranks all documents based on entity coverage and claim alignment

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()

        # Retrieval modules with different k values
        self.retrieve_15 = dspy.Retrieve(k=15)  # Hop 1: broader initial retrieval
        self.retrieve_10 = dspy.Retrieve(k=10)  # Hops 2-3: targeted retrieval

        # Entity extraction and coverage analysis
        self.extract_entities = dspy.ChainOfThought(ExtractClaimEntities)
        self.verify_coverage = dspy.ChainOfThought(VerifyEntityCoverage)

        # Query generation for uncovered entities
        self.create_entity_query = dspy.ChainOfThought("claim,entity,context->query")

        # Document reranking
        self.rank_documents = dspy.ChainOfThought(RankDocumentsByRelevance)

    def forward(self, claim):
        # Step 1: Extract entities from the claim
        entity_extraction = self.extract_entities(claim=claim)
        entities = entity_extraction.entities

        # Step 2: HOP 1 - Retrieve k=15 documents with full claim query
        hop1_docs = self.retrieve_15(claim).passages

        # Step 3: Verify entity coverage and identify gaps
        coverage_analysis = self.verify_coverage(
            claim=claim,
            entities=entities,
            documents=hop1_docs
        )
        uncovered_entities = coverage_analysis.uncovered_entities

        # Step 4: HOP 2 - Retrieve k=10 documents for first uncovered entity
        hop2_docs = []
        if len(uncovered_entities) > 0:
            entity_1 = uncovered_entities[0]
            hop2_query = self.create_entity_query(
                claim=claim,
                entity=entity_1,
                context=f"Retrieved documents mention: {', '.join(entities[:3])}"
            ).query
            hop2_docs = self.retrieve_10(hop2_query).passages

        # Step 5: HOP 3 - Retrieve k=10 documents for second uncovered entity
        hop3_docs = []
        if len(uncovered_entities) > 1:
            entity_2 = uncovered_entities[1]
            hop3_query = self.create_entity_query(
                claim=claim,
                entity=entity_2,
                context=f"Previously searched for: {entity_1 if len(uncovered_entities) > 0 else 'initial entities'}"
            ).query
            hop3_docs = self.retrieve_10(hop3_query).passages

        # Step 6: Combine all retrieved documents (15 + 10 + 10 = 35 documents)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Step 7: Remove duplicates while preserving order
        unique_docs = []
        seen = set()
        for doc in all_docs:
            # Use a simple hash of the document content to detect duplicates
            doc_hash = hash(doc[:200] if len(doc) > 200 else doc)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        # Step 8: Rerank documents based on entity coverage and claim alignment
        if len(unique_docs) > 21:
            ranking = self.rank_documents(
                claim=claim,
                entities=entities,
                documents=unique_docs
            )
            relevance_scores = ranking.relevance_scores

            # Pair documents with scores and sort by relevance
            doc_score_pairs = list(zip(unique_docs, relevance_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Select top 21 documents
            final_docs = [doc for doc, score in doc_score_pairs[:21]]
        else:
            final_docs = unique_docs[:21]

        return dspy.Prediction(
            retrieved_docs=final_docs,
            entities=entities,
            uncovered_entities=uncovered_entities
        )
