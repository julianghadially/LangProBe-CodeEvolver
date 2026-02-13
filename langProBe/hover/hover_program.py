import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate


class ExtractEntitiesSignature(dspy.Signature):
    """
    Extract the 2-3 most important entities or concepts from the claim that require verification.
    Entities can be people, places, organizations, events, works of art, or specific concepts.
    These entities will be used to guide the retrieval process for fact-checking.
    Focus on named entities that are central to verifying the claim's truthfulness.
    """

    claim = dspy.InputField(desc="The claim to be verified")
    entities: list[str] = dspy.OutputField(
        desc="A list of 2-3 key entities or concepts from the claim, ordered by importance"
    )


class GeneratePrimaryEntityQuerySignature(dspy.Signature):
    """
    Generate a search query to retrieve information about the primary entity in the claim.
    Focus on finding the main Wikipedia article or definitive source about this entity.
    The query should be direct and specific to retrieve the entity's main page.
    """

    claim = dspy.InputField(desc="The claim to be verified")
    primary_entity = dspy.InputField(desc="The most important entity from the claim")
    search_query = dspy.OutputField(
        desc="A search query targeting the primary entity"
    )


class AnalyzeRetrievalGapsSignature(dspy.Signature):
    """
    Analyze the retrieved documents against the claim to identify what critical information is still missing.
    Determine what additional facts, relationships, or context need to be retrieved to fully verify the claim.
    Consider what aspects of the claim have not been addressed by the documents retrieved so far.
    """

    claim = dspy.InputField(desc="The claim to be verified")
    entities: list[str] = dspy.InputField(desc="The entities extracted from the claim")
    context: list[str] = dspy.InputField(desc="Documents retrieved so far")
    missing_info = dspy.OutputField(
        desc="Description of the critical information that is still missing to verify the claim"
    )


class GenerateGapQuerySignature(dspy.Signature):
    """
    Generate a search query to find the missing information identified in the gap analysis.
    The query should target specific facts or relationships needed to verify the claim.
    Build on what has already been retrieved to avoid redundancy.
    """

    claim = dspy.InputField(desc="The claim to be verified")
    missing_info = dspy.InputField(desc="The critical information that is missing")
    context: list[str] = dspy.InputField(desc="Documents retrieved so far")
    search_query = dspy.OutputField(
        desc="A search query targeting the missing information"
    )


class GenerateRelatedConceptsQuerySignature(dspy.Signature):
    """
    Generate a search query to retrieve information about related concepts or entities not yet covered.
    This query fills in final gaps and provides additional context for verification.
    Look for secondary entities, related events, or contextual information.
    """

    claim = dspy.InputField(desc="The claim to be verified")
    entities: list[str] = dspy.InputField(desc="All entities from the claim")
    context: list[str] = dspy.InputField(desc="Documents retrieved so far")
    search_query = dspy.OutputField(
        desc="A search query for related concepts or secondary entities"
    )


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        # Retrieval parameters
        self.k = 7  # Documents per hop
        self.max_hops = 3  # Total retrieval hops

        # Step 1: Entity Extraction (using ChainOfThought for reasoning)
        self.extract_entities = dspy.ChainOfThought(ExtractEntitiesSignature)

        # Step 2: First Retrieval - Primary Entity Query
        self.generate_primary_query = dspy.ChainOfThought(
            GeneratePrimaryEntityQuerySignature
        )
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Step 3: Gap Analysis
        self.analyze_gaps = dspy.ChainOfThought(AnalyzeRetrievalGapsSignature)

        # Step 4: Second Retrieval - Gap-based Query
        self.generate_gap_query = dspy.ChainOfThought(GenerateGapQuerySignature)

        # Step 5: Third Retrieval - Related Concepts Query
        self.generate_related_query = dspy.ChainOfThought(
            GenerateRelatedConceptsQuerySignature
        )

    def forward(self, claim):
        """
        Execute entity-aware sequential retrieval with three hops.

        Args:
            claim: The claim to verify

        Returns:
            dspy.Prediction with retrieved_docs containing all deduplicated documents (up to 21)
        """

        # Step 1: Extract Entities
        entities_pred = self.extract_entities(claim=claim)
        entities = entities_pred.entities
        primary_entity = entities[0] if entities else ""

        # Initialize context list for accumulating documents
        context = []

        # Step 2: First Hop - Primary Entity Retrieval
        query1_pred = self.generate_primary_query(
            claim=claim, primary_entity=primary_entity
        )
        query1 = query1_pred.search_query
        passages1 = self.retrieve_k(query1).passages
        context = deduplicate(context + passages1)

        # Step 3: Gap Analysis
        gaps_pred = self.analyze_gaps(claim=claim, entities=entities, context=context)
        missing_info = gaps_pred.missing_info

        # Step 4: Second Hop - Gap-based Retrieval
        query2_pred = self.generate_gap_query(
            claim=claim, missing_info=missing_info, context=context
        )
        query2 = query2_pred.search_query
        passages2 = self.retrieve_k(query2).passages
        context = deduplicate(context + passages2)

        # Step 5: Third Hop - Related Concepts Retrieval
        query3_pred = self.generate_related_query(
            claim=claim, entities=entities, context=context
        )
        query3 = query3_pred.search_query
        passages3 = self.retrieve_k(query3).passages
        context = deduplicate(context + passages3)

        # Return all documents (deduplicated, max 21)
        return dspy.Prediction(retrieved_docs=context)


