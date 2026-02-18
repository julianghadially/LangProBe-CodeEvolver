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


class GapAnalysisModule(dspy.Module):
    """Module for gap-aware entity tracking and targeted query generation."""

    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(ExtractEntitiesSignature)
        self.analyze_coverage = dspy.ChainOfThought(AnalyzeCoverageSignature)
        self.generate_gap_query = dspy.ChainOfThought(GenerateGapQuerySignature)


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7

        # NEW: Gap analysis module
        self.gap_analyzer = GapAnalysisModule()

        # EXISTING: Keep for backward compatibility and fallback
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # STEP 1: Extract required entities from claim (once, at the start)
        entity_extraction = self.gap_analyzer.extract_entities(claim=claim)
        required_entities = entity_extraction.entities

        # Handle edge case: no entities extracted
        if not required_entities:
            required_entities = [claim]  # Treat entire claim as single entity

        # HOP 1: Initial retrieval based on claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

        # STEP 2: Analyze coverage after hop 1
        coverage_hop1 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=hop1_docs
        )

        # HOP 2: Generate gap-targeted query or fallback to original
        if coverage_hop1.missing_entities:
            hop2_query = self.gap_analyzer.generate_gap_query(
                claim=claim,
                missing_entities=coverage_hop1.missing_entities,
                coverage_summary=coverage_hop1.coverage_summary,
                context=summary_1
            ).query
        else:
            # Fallback: use existing query generator if all entities covered
            hop2_query = self.create_query_hop2(
                claim=claim, summary_1=summary_1
            ).query

        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # STEP 3: Analyze coverage after hop 2 (on all passages so far)
        all_passages_so_far = hop1_docs + hop2_docs
        coverage_hop2 = self.gap_analyzer.analyze_coverage(
            claim=claim,
            required_entities=required_entities,
            passages=all_passages_so_far
        )

        # HOP 3: Generate gap-targeted query for remaining gaps or fallback
        if coverage_hop2.missing_entities:
            hop3_query = self.gap_analyzer.generate_gap_query(
                claim=claim,
                missing_entities=coverage_hop2.missing_entities,
                coverage_summary=coverage_hop2.coverage_summary,
                context=summary_1 + " " + summary_2
            ).query
        else:
            # Fallback: broaden search if all entities covered
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query

        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
