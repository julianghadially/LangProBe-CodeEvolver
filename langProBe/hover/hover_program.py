import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractionSignature(dspy.Signature):
    """Extract 3-5 key named entities (people, places, things, titles) from the claim that are critical to verification.
    Focus on entities that would need evidence to verify the claim."""

    claim = dspy.InputField()
    entities: list[str] = dspy.OutputField(
        desc="List of 3-5 key named entities (people, places, organizations, titles, events) critical for verifying the claim"
    )


class GapAnalysisSignature(dspy.Signature):
    """Identify which entities from the original list are NOT well-covered in the retrieved passages.
    Generate a targeted query focusing on those missing entities to fill the information gaps."""

    claim = dspy.InputField()
    entities = dspy.InputField(desc="Original list of key entities that need to be verified")
    passages = dspy.InputField(desc="Retrieved passages from previous hop")
    missing_entities: list[str] = dspy.OutputField(
        desc="Entities from the original list that are NOT well-covered in the passages"
    )
    next_query: str = dspy.OutputField(
        desc="Targeted search query focusing on the missing entities to fill information gaps"
    )


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction and gap analysis modules
        self.extract_entities = dspy.ChainOfThought(EntityExtractionSignature)
        self.gap_analysis_hop2 = dspy.ChainOfThought(GapAnalysisSignature)
        self.gap_analysis_hop3 = dspy.ChainOfThought(GapAnalysisSignature)

        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # STEP 1: Extract key entities from the claim
        entity_extraction = self.extract_entities(claim=claim)
        entities = entity_extraction.entities

        # HOP 1: Construct query combining claim with entities
        hop1_query = f"{claim} {' '.join(entities)}"
        hop1_docs = self.retrieve_k(hop1_query).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # STEP 2: Gap analysis after hop1 - identify missing entities
        gap_analysis_2 = self.gap_analysis_hop2(
            claim=claim,
            entities=entities,
            passages=hop1_docs
        )
        missing_entities_2 = gap_analysis_2.missing_entities
        hop2_query = gap_analysis_2.next_query

        # HOP 2: Retrieve with missing-entity-focused query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # STEP 3: Gap analysis after hop2 - identify remaining uncovered entities
        gap_analysis_3 = self.gap_analysis_hop3(
            claim=claim,
            entities=missing_entities_2,  # Focus on entities still missing after hop2
            passages=hop2_docs
        )
        hop3_query = gap_analysis_3.next_query

        # HOP 3: Retrieve with query targeting remaining uncovered entities
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


