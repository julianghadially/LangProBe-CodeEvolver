import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractEntitiesSignature(dspy.Signature):
    """
    Extract the key entities, concepts, and relationships that must be verified to
    evaluate the claim. Focus on named entities (people, organizations, locations,
    events) and specific facts that require evidence. Limit to the 10 most important
    entities, prioritizing named entities and specific facts over general concepts.
    """

    claim = dspy.InputField(desc="The claim to analyze")
    entities: list[str] = dspy.OutputField(
        desc="List of key entities, concepts, and facts that need evidence. "
        "Each entity should be specific and searchable (e.g., 'Barack Obama', "
        "'2008 presidential election', 'Nobel Peace Prize 2009')"
    )


class AnalyzeCoverageSignature(dspy.Signature):
    """
    Analyze which entities and concepts are covered by the retrieved passages
    and which are still missing. An entity is 'covered' if there is substantive
    information about it in the passages, not just a passing mention.
    """

    claim = dspy.InputField(desc="The original claim")
    required_entities = dspy.InputField(
        desc="List of entities that need to be covered"
    )
    passages = dspy.InputField(
        desc="Retrieved passages from previous hop(s)"
    )
    covered_entities: list[str] = dspy.OutputField(
        desc="Entities that have substantive coverage in the passages"
    )
    missing_entities: list[str] = dspy.OutputField(
        desc="Entities that are not covered or only mentioned in passing"
    )
    coverage_summary = dspy.OutputField(
        desc="Brief summary of what information is present and what gaps remain"
    )


class GenerateGapQuerySignature(dspy.Signature):
    """
    Generate a search query specifically targeting the missing entities and information
    gaps. The query should be focused and likely to retrieve documents that address
    the uncovered aspects of the claim.
    """

    claim = dspy.InputField(desc="The original claim")
    missing_entities = dspy.InputField(
        desc="Entities and concepts that are not yet covered"
    )
    coverage_summary = dspy.InputField(
        desc="Summary of what information is present and what gaps remain"
    )
    context = dspy.InputField(
        desc="Information already gathered from previous hops",
        default=""
    )
    query = dspy.OutputField(
        desc="Search query targeting the missing information. Should be specific "
        "and focused on retrieving documents about the uncovered entities."
    )


class GenerateMultipleQueriesSignature(dspy.Signature):
    """
    Generate 2-3 diverse search queries to retrieve documents addressing different
    aspects of the claim. Each query should target different missing entities or
    information gaps. Queries should be complementary, not redundant.
    """

    claim = dspy.InputField(desc="The original claim to verify")
    missing_entities = dspy.InputField(
        desc="Entities and concepts not yet covered in retrieved documents"
    )
    coverage_summary = dspy.InputField(
        desc="Summary of what information is present and what gaps remain"
    )
    context = dspy.InputField(
        desc="Information already gathered from previous hops",
        default=""
    )
    queries: list[str] = dspy.OutputField(
        desc="List of 2-3 diverse search queries. Each query should target different "
        "entities or aspects from missing_entities. Queries must be complementary and "
        "non-overlapping to maximize coverage."
    )


class UtilityRerankerSignature(dspy.Signature):
    """
    Score a document's utility for verifying the claim based on three criteria:
    1. Relevance: How relevant is the document to the claim?
    2. Entity Coverage: Does it cover uncovered entities from required_entities?
    3. Diversity: Does it provide new information compared to already_selected_docs?

    Higher scores indicate higher utility. Score should be between 0.0 and 10.0.
    """

    claim = dspy.InputField(desc="The claim being verified")
    document = dspy.InputField(
        desc="The document to score, in format '[TITLE] | [CONTENT]'"
    )
    required_entities = dspy.InputField(
        desc="List of entities that need to be covered to verify the claim"
    )
    covered_entities = dspy.InputField(
        desc="Entities already covered by previously selected documents"
    )
    already_selected_docs = dspy.InputField(
        desc="Titles of documents already selected (to assess diversity)",
        default=""
    )
    utility_score: float = dspy.OutputField(
        desc="Utility score from 0.0 to 10.0. Higher scores indicate higher utility "
        "considering relevance, entity coverage, and diversity."
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of the score, covering relevance, entity coverage, "
        "and diversity aspects."
    )


class GapAnalysisModule(dspy.Module):
    """Module for gap-aware entity tracking and targeted query generation."""

    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(ExtractEntitiesSignature)
        self.analyze_coverage = dspy.ChainOfThought(AnalyzeCoverageSignature)
        self.generate_gap_query = dspy.ChainOfThought(GenerateGapQuerySignature)


class MultiQueryGenerator(dspy.Module):
    """Generates multiple diverse queries for a single hop."""

    def __init__(self):
        super().__init__()
        self.generate_queries = dspy.ChainOfThought(GenerateMultipleQueriesSignature)

    def forward(self, claim, missing_entities, coverage_summary, context=""):
        result = self.generate_queries(
            claim=claim,
            missing_entities=missing_entities,
            coverage_summary=coverage_summary,
            context=context
        )
        # Ensure we get 2-3 queries, handle edge cases
        queries = result.queries if isinstance(result.queries, list) else [result.queries]
        # Limit to 3 queries maximum, minimum 1
        if len(queries) == 0:
            queries = [claim]  # Fallback to claim as query
        return queries[:3]


class UtilityReranker(dspy.Module):
    """Scores documents based on utility for claim verification."""

    def __init__(self):
        super().__init__()
        self.score_document = dspy.ChainOfThought(UtilityRerankerSignature)

    def forward(self, claim, document, required_entities, covered_entities,
                already_selected_docs=""):
        result = self.score_document(
            claim=claim,
            document=document,
            required_entities=required_entities,
            covered_entities=covered_entities,
            already_selected_docs=already_selected_docs
        )
        # Parse score, handle edge cases
        try:
            score = float(result.utility_score)
            # Clamp to valid range
            score = max(0.0, min(10.0, score))
        except (ValueError, TypeError):
            # Fallback: default to 5.0
            score = 5.0

        return score, result.reasoning

    def batch_score(self, claim, documents, required_entities, covered_entities,
                    already_selected_titles):
        """Score multiple documents efficiently."""
        scores = []
        selected_titles_str = ", ".join(already_selected_titles) if already_selected_titles else ""

        for doc in documents:
            score, reasoning = self.forward(
                claim=claim,
                document=doc,
                required_entities=required_entities,
                covered_entities=covered_entities,
                already_selected_docs=selected_titles_str
            )
            scores.append((doc, score, reasoning))

        return scores


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 70  # CHANGED: Increased from 7 to 70 for better recall

        # NEW: Multi-query and reranking modules
        self.multi_query_generator = MultiQueryGenerator()
        self.utility_reranker = UtilityReranker()

        # EXISTING: Gap analysis module
        self.gap_analyzer = GapAnalysisModule()

        # EXISTING: Keep for backward compatibility and fallback
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _deduplicate_by_title(self, documents):
        """
        Deduplicate documents by title (before ' | ' separator).
        Preserves order of first occurrence.
        """
        seen_titles = set()
        unique_docs = []

        for doc in documents:
            # Extract title
            title = doc.split(" | ")[0] if " | " in doc else doc

            # Add if not seen
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

        return unique_docs

    def forward(self, claim):
        # STEP 1: Extract required entities from claim (once at start)
        entity_extraction = self.gap_analyzer.extract_entities(claim=claim)
        required_entities = entity_extraction.entities

        if not required_entities:
            required_entities = [claim]

        # Container for all retrieved documents across all hops
        all_retrieved_docs = []

        # Track summaries for context building
        summaries = []

        # === HOP 1: Multi-query retrieval ===
        # Initial coverage analysis (no docs yet, all entities missing)
        initial_coverage = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=[]
        )

        # Generate multiple diverse queries for hop 1
        hop1_queries = self.multi_query_generator(
            claim=claim,
            missing_entities=initial_coverage.missing_entities or required_entities,
            coverage_summary=initial_coverage.coverage_summary or "No information retrieved yet",
            context=""
        )

        # Retrieve documents for each query (k=70 per query)
        hop1_docs = []
        for query in hop1_queries:
            docs = self.retrieve_k(query).passages
            hop1_docs.extend(docs)

        all_retrieved_docs.extend(hop1_docs)

        # Summarize hop 1 results (use deduplicated subset for efficiency)
        hop1_docs_dedup = self._deduplicate_by_title(hop1_docs)[:21]
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs_dedup).summary
        summaries.append(summary_1)

        # === HOP 2: Gap-aware multi-query retrieval ===
        # Analyze coverage after hop 1
        coverage_hop1 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=hop1_docs_dedup
        )

        # Generate diverse queries targeting gaps
        if coverage_hop1.missing_entities:
            hop2_queries = self.multi_query_generator(
                claim=claim,
                missing_entities=coverage_hop1.missing_entities,
                coverage_summary=coverage_hop1.coverage_summary,
                context=summary_1
            )
        else:
            # Fallback if no gaps: generate single broader query
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_queries = [hop2_query]

        # Retrieve for each query (k=70 per query)
        hop2_docs = []
        for query in hop2_queries:
            docs = self.retrieve_k(query).passages
            hop2_docs.extend(docs)

        all_retrieved_docs.extend(hop2_docs)

        # Summarize hop 2
        hop2_docs_dedup = self._deduplicate_by_title(hop2_docs)[:21]
        summary_2 = self.summarize2(
            claim=claim,
            context=summary_1,
            passages=hop2_docs_dedup
        ).summary
        summaries.append(summary_2)

        # === HOP 3: Final gap-targeted multi-query retrieval ===
        # Analyze coverage after hop 2 (use combined docs from both hops)
        all_docs_so_far = self._deduplicate_by_title(hop1_docs + hop2_docs)[:42]
        coverage_hop2 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=all_docs_so_far
        )

        # Generate queries for remaining gaps
        if coverage_hop2.missing_entities:
            hop3_queries = self.multi_query_generator(
                claim=claim,
                missing_entities=coverage_hop2.missing_entities,
                coverage_summary=coverage_hop2.coverage_summary,
                context=" ".join(summaries)
            )
        else:
            # Fallback: broaden search
            hop3_query = self.create_query_hop3(
                claim=claim,
                summary_1=summary_1,
                summary_2=summary_2
            ).query
            hop3_queries = [hop3_query]

        # Retrieve for each query (k=70 per query)
        hop3_docs = []
        for query in hop3_queries:
            docs = self.retrieve_k(query).passages
            hop3_docs.extend(docs)

        all_retrieved_docs.extend(hop3_docs)

        # === STEP 2: Deduplicate all documents by title ===
        unique_docs = self._deduplicate_by_title(all_retrieved_docs)

        # === STEP 3: Utility-based reranking ===
        # Analyze final coverage for reranking context
        final_coverage = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=unique_docs[:50]  # Sample for efficiency
        )

        # Score all unique documents
        scored_docs = self.utility_reranker.batch_score(
            claim=claim,
            documents=unique_docs,
            required_entities=required_entities,
            covered_entities=final_coverage.covered_entities,
            already_selected_titles=[]  # Empty for batch scoring
        )

        # Sort by score (descending) and select top 21
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_21_docs = [doc for doc, score, reasoning in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
