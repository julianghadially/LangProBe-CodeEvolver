import json
import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


def deduplicate_passages(passages: list[str]) -> list[str]:
    """Deduplicate passages by document key (first part before ' | ').

    Args:
        passages: List of passages in format "doc_key | passage_text"

    Returns:
        Deduplicated list preserving order
    """
    seen_keys = set()
    result = []
    for passage in passages:
        key = passage.split(" | ")[0] if " | " in passage else passage
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(passage)
    return result


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


class EntityExtractionSignature(dspy.Signature):
    """Extract 2-4 key named entities from a claim that are essential for fact-checking.
    Focus on proper nouns (people, places, organizations, events) that require verification."""

    claim = dspy.InputField(desc="A claim to fact-check")
    entities = dspy.OutputField(
        desc="List of 2-4 key named entities as comma-separated values (e.g., 'Barack Obama, United States, 2008 election')"
    )


class RelationshipQuerySignature(dspy.Signature):
    """Generate a search query to find documents about relationships between entities.
    Focus on connections, interactions, or shared attributes between the known entities."""

    claim = dspy.InputField(desc="The original claim to verify")
    known_entities = dspy.InputField(desc="Entities already identified and their context from Stage 1")
    query = dspy.OutputField(desc="A search query focused on relationships between entities")


class HoverMultiHopCascading(LangProBeDSPyMetaProgram, dspy.Module):
    """2-stage cascading retrieval for HoverMultiHop.

    Stage 1: Entity-focused retrieval (7 docs)
        - Extract 2-4 entities
        - Retrieve k=3 per entity (6-12 docs)
        - Deduplicate and select top-7

    Stage 2: Relationship-focused 2-hop retrieval (14 docs)
        - Generate relationship queries
        - Hop 1: k=7 docs
        - Hop 2: k=7 docs

    Total: 21 documents

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Entity extraction and retrieval
        self.extract_entities = dspy.ChainOfThought(EntityExtractionSignature)
        self.retrieve_k3 = dspy.Retrieve(k=3)
        self.summarize_stage1 = dspy.ChainOfThought(
            "claim, entities, passages -> summary"
        )

        # Stage 2: Relationship-focused retrieval
        self.generate_relationship_query = dspy.ChainOfThought(
            RelationshipQuerySignature
        )
        self.retrieve_k7 = dspy.Retrieve(k=7)
        self.summarize_hop1 = dspy.ChainOfThought(
            "claim, context, passages -> summary"
        )
        self.generate_hop2_query = dspy.ChainOfThought(
            "claim, known_entities, hop1_context -> query"
        )

    def parse_entities(self, entities_str: str) -> list[str]:
        """Parse entity string into list, handling various formats.

        Args:
            entities_str: Comma-separated entity string or JSON array

        Returns:
            List of entity strings (max 4)
        """
        # Try JSON parsing first
        try:
            entities = json.loads(entities_str)
            if isinstance(entities, list):
                return [str(e) for e in entities[:4]]
        except:
            pass

        # Fallback: comma-separated parsing
        entities = [e.strip() for e in entities_str.split(",")]
        return entities[:4]

    def forward(self, claim):
        """Execute 2-stage cascading retrieval.

        Args:
            claim: The claim to fact-check

        Returns:
            dspy.Prediction with retrieved_docs (21 documents)
        """
        # STAGE 1: Entity-Focused Retrieval
        # Step 1: Extract entities
        entity_extraction = self.extract_entities(claim=claim)
        entities = self.parse_entities(entity_extraction.entities)

        # Step 2: Retrieve for each entity (k=3 per entity)
        stage1_passages = []
        for entity in entities:
            try:
                passages = self.retrieve_k3(entity).passages
                stage1_passages.extend(passages)
            except Exception as e:
                # Continue if retrieval fails for an entity
                continue

        # Step 3: Deduplicate and select top-7
        stage1_passages = deduplicate_passages(stage1_passages)[:7]

        # If we have fewer than 7, retrieve more with the full claim
        if len(stage1_passages) < 7:
            extra_docs = self.retrieve_k7(claim).passages
            stage1_passages.extend(extra_docs)
            stage1_passages = deduplicate_passages(stage1_passages)[:7]

        # Step 4: Summarize Stage 1 findings
        entity_summary = self.summarize_stage1(
            claim=claim,
            entities=", ".join(entities),
            passages=stage1_passages
        ).summary

        # STAGE 2: Relationship-Focused Retrieval
        # Step 5: Generate relationship query for Hop 1
        hop1_query = self.generate_relationship_query(
            claim=claim,
            known_entities=entity_summary
        ).query
        hop1_docs = self.retrieve_k7(hop1_query).passages

        # Step 6: Summarize Hop 1 and generate Hop 2 query
        hop1_summary = self.summarize_hop1(
            claim=claim,
            context=entity_summary,
            passages=hop1_docs
        ).summary

        hop2_query = self.generate_hop2_query(
            claim=claim,
            known_entities=entity_summary,
            hop1_context=hop1_summary
        ).query
        hop2_docs = self.retrieve_k7(hop2_query).passages

        # Step 7: Deduplicate Stage 2 and limit to 14
        stage2_passages = deduplicate_passages(hop1_docs + hop2_docs)[:14]

        # Final combination: exactly 21 documents
        final_docs = stage1_passages + stage2_passages
        return dspy.Prediction(retrieved_docs=final_docs[:21])
