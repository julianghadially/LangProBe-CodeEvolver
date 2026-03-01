import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractor(dspy.Signature):
    """Extract 2-3 key entities or topics from the claim that need supporting documents for verification."""
    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="2-3 key entities/topics that need supporting documents (e.g., person names, book titles, organizations)")


class CoverageAnalyzer(dspy.Signature):
    """Analyze which entities are well-covered by current documents and which need more retrieval."""
    claim: str = dspy.InputField()
    entities: str = dspy.InputField(desc="comma-separated list of entities to verify")
    current_documents: str = dspy.InputField(desc="titles of currently retrieved documents")
    well_covered_entities: str = dspy.OutputField(desc="comma-separated entities that have sufficient document coverage")
    under_covered_entities: str = dspy.OutputField(desc="comma-separated entities that need more document retrieval")
    confidence_score: float = dspy.OutputField(desc="confidence score (0-1) for overall coverage of all entities")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the claim on a scale of 0-10."""
    claim: str = dspy.InputField()
    document_title: str = dspy.InputField()
    document_content: str = dspy.InputField()
    relevance_score: float = dspy.OutputField(desc="relevance score from 0 (irrelevant) to 10 (highly relevant)")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25  # Increased from 7 to 25 per query
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # New modules for confidence-weighted retrieval
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)
        self.coverage_analyzer = dspy.ChainOfThought(CoverageAnalyzer)
        self.relevance_scorer = dspy.ChainOfThought(RelevanceScorer)

        # Query generator for under-covered entities
        self.generate_query = dspy.ChainOfThought("claim, entity -> query")

    def forward(self, claim):
        # Step 1: Extract key entities from the claim
        entity_result = self.entity_extractor(claim=claim)
        entities = entity_result.entities if isinstance(entity_result.entities, list) else []

        # Handle case where entities might be returned as a string
        if not entities and hasattr(entity_result, 'entities'):
            entities_str = str(entity_result.entities)
            # Try to parse comma-separated or newline-separated entities
            if ',' in entities_str:
                entities = [e.strip() for e in entities_str.split(',')]
            elif '\n' in entities_str:
                entities = [e.strip() for e in entities_str.split('\n') if e.strip()]
            else:
                entities = [entities_str.strip()]

        # Ensure we have 2-3 entities
        entities = [e for e in entities if e][:3]
        if not entities:
            entities = [claim]  # Fallback to using claim itself

        # Step 2: Initialize deduplication set and document collection
        seen_titles = set()
        all_documents = []

        # Step 3: Perform 3 hops with coverage-based retrieval
        max_queries_per_hop = 3
        num_hops = 3

        for hop in range(num_hops):
            # Get current document titles for coverage analysis
            current_doc_titles = ", ".join(seen_titles) if seen_titles else "None"
            entities_str = ", ".join(entities)

            # Analyze coverage to identify under-covered entities
            coverage_result = self.coverage_analyzer(
                claim=claim,
                entities=entities_str,
                current_documents=current_doc_titles
            )

            # Parse under-covered entities
            under_covered = coverage_result.under_covered_entities
            if isinstance(under_covered, str):
                under_covered_list = [e.strip() for e in under_covered.split(',') if e.strip()]
            else:
                under_covered_list = list(under_covered) if under_covered else []

            # If no under-covered entities identified, use all entities (fallback)
            if not under_covered_list:
                under_covered_list = entities

            # Limit to max queries per hop
            entities_to_query = under_covered_list[:max_queries_per_hop]

            # Generate queries for under-covered entities and retrieve documents
            for entity in entities_to_query:
                # Generate focused query for this entity
                query_result = self.generate_query(claim=claim, entity=entity)
                query = query_result.query

                # Retrieve documents
                retrieved = self.retrieve_k(query)
                passages = retrieved.passages if hasattr(retrieved, 'passages') else []

                # Deduplicate: only add documents not already seen
                for doc in passages:
                    # Extract document title (first part before " | ")
                    doc_title = doc.split(" | ")[0] if " | " in doc else doc

                    # Normalize title for comparison
                    normalized_title = doc_title.strip().lower()

                    if normalized_title not in seen_titles:
                        seen_titles.add(normalized_title)
                        all_documents.append(doc)

        # Step 4: Rerank all unique documents to select top 21
        scored_docs = []
        for doc in all_documents:
            # Split document into title and content
            if " | " in doc:
                doc_title, doc_content = doc.split(" | ", 1)
            else:
                doc_title = doc
                doc_content = ""

            # Score document relevance
            try:
                score_result = self.relevance_scorer(
                    claim=claim,
                    document_title=doc_title,
                    document_content=doc_content[:500]  # Limit content length
                )
                score = float(score_result.relevance_score)
            except (ValueError, AttributeError):
                score = 5.0  # Default middle score if parsing fails

            scored_docs.append((score, doc))

        # Sort by score descending and take top 21
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_docs)
