import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractEntities(dspy.Signature):
    """Extract key entities from the claim and context.

    Focus on identifying:
    - People names (full names, aliases)
    - Places (cities, countries, locations)
    - Organizations (companies, institutions, groups)
    - Titles (movies, books, songs, TV shows, albums)
    - Dates and time periods
    - Other proper nouns critical to the claim

    These entities are essential for multi-hop retrieval to preserve critical
    information that may be lost during summarization."""

    claim = dspy.InputField(desc="The claim to verify")
    context = dspy.InputField(desc="Additional context from retrieved passages")
    entities: list[str] = dspy.OutputField(
        desc="List of key entities (people, places, titles, organizations, dates) extracted from claim and context. Include full names and proper nouns."
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    APPROACH
    - Uses explicit entity extraction to preserve critical information across retrieval hops
    - Entities (people, places, titles, organizations) are tracked and used to create focused queries
    - This prevents information loss that occurs with pure summarization approaches
    '''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Entity extraction modules
        self.extract_entities_initial = dspy.ChainOfThought(ExtractEntities)
        self.extract_entities_hop1 = dspy.ChainOfThought(ExtractEntities)
        self.extract_entities_hop2 = dspy.ChainOfThought(ExtractEntities)

        # Query generation modules with entity-focused context
        self.create_query_hop2 = dspy.ChainOfThought("claim, entities->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, entities->query")

    def forward(self, claim):
        # STEP 1: Extract entities from the claim before hop 1
        initial_entities_pred = self.extract_entities_initial(
            claim=claim,
            context=""
        )
        entities_from_claim = initial_entities_pred.entities

        # HOP 1: Initial retrieval using the original claim
        hop1_docs = self.retrieve_k(claim).passages

        # STEP 2: Extract new entities from hop 1 passages
        hop1_context = "\n".join(hop1_docs[:3])  # Use top 3 passages for entity extraction
        hop1_entities_pred = self.extract_entities_hop1(
            claim=claim,
            context=hop1_context
        )
        entities_from_hop1 = hop1_entities_pred.entities

        # STEP 3: Create hop 2 query using claim + entities from claim and hop 1
        all_entities_hop1 = list(set(entities_from_claim + entities_from_hop1))
        entities_str_hop1 = ", ".join(all_entities_hop1)

        hop2_query = self.create_query_hop2(
            claim=claim,
            entities=entities_str_hop1
        ).query

        # HOP 2: Retrieval with entity-enriched query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # STEP 4: Extract new entities from hop 2 passages
        hop2_context = "\n".join(hop2_docs[:3])  # Use top 3 passages for entity extraction
        hop2_entities_pred = self.extract_entities_hop2(
            claim=claim,
            context=hop2_context
        )
        entities_from_hop2 = hop2_entities_pred.entities

        # STEP 5: Create hop 3 query using claim + all extracted entities from hops 1 and 2
        all_entities_hop2 = list(set(all_entities_hop1 + entities_from_hop2))
        entities_str_hop2 = ", ".join(all_entities_hop2)

        hop3_query = self.create_query_hop3(
            claim=claim,
            entities=entities_str_hop2
        ).query

        # HOP 3: Final retrieval with all accumulated entities
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
