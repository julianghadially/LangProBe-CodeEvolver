import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class MultiAspectExtraction(dspy.Signature):
    """Extract structured information from passages to preserve retrieval-relevant details.
    Identify key entities (people, places, organizations, dates), verifiable facts,
    and potential connections to related documents."""

    claim = dspy.InputField(desc="The claim being verified")
    passages = dspy.InputField(desc="Retrieved passages to analyze")
    key_entities: list[str] = dspy.OutputField(
        desc="List of important entities mentioned (names, places, organizations, dates, etc.)"
    )
    key_facts: list[str] = dspy.OutputField(
        desc="List of key verifiable facts that could support or refute the claim"
    )
    connections: str = dspy.OutputField(
        desc="Potential connections or references to other related documents, topics, or entities"
    )


class MultiAspectExtractionHop2(dspy.Signature):
    """Extract structured information from second hop passages, building on previous context.
    Identify key entities, verifiable facts, and connections that bridge to additional sources."""

    claim = dspy.InputField(desc="The claim being verified")
    context = dspy.InputField(desc="Context from previous hop")
    passages = dspy.InputField(desc="Retrieved passages to analyze")
    key_entities: list[str] = dspy.OutputField(
        desc="List of important entities mentioned (names, places, organizations, dates, etc.)"
    )
    key_facts: list[str] = dspy.OutputField(
        desc="List of key verifiable facts that could support or refute the claim"
    )
    connections: str = dspy.OutputField(
        desc="Potential connections or references to other related documents, topics, or entities"
    )


class QueryGenerationHop2(dspy.Signature):
    """Generate a search query for the second hop using structured information from the first hop."""

    claim = dspy.InputField(desc="The claim being verified")
    key_entities_1: list[str] = dspy.InputField(desc="Key entities from hop 1")
    key_facts_1: list[str] = dspy.InputField(desc="Key facts from hop 1")
    connections_1: str = dspy.InputField(desc="Potential connections from hop 1")
    query: str = dspy.OutputField(desc="Search query for second hop retrieval")


class QueryGenerationHop3(dspy.Signature):
    """Generate a search query for the third hop using structured information from both previous hops."""

    claim = dspy.InputField(desc="The claim being verified")
    key_entities_1: list[str] = dspy.InputField(desc="Key entities from hop 1")
    key_facts_1: list[str] = dspy.InputField(desc="Key facts from hop 1")
    connections_1: str = dspy.InputField(desc="Potential connections from hop 1")
    key_entities_2: list[str] = dspy.InputField(desc="Key entities from hop 2")
    key_facts_2: list[str] = dspy.InputField(desc="Key facts from hop 2")
    connections_2: str = dspy.InputField(desc="Potential connections from hop 2")
    query: str = dspy.OutputField(desc="Search query for third hop retrieval")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_hop1 = dspy.ChainOfThought(MultiAspectExtraction)
        self.extract_hop2 = dspy.ChainOfThought(MultiAspectExtractionHop2)
        self.create_query_hop2 = dspy.ChainOfThought(QueryGenerationHop2)
        self.create_query_hop3 = dspy.ChainOfThought(QueryGenerationHop3)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Extract structured information from initial retrieval
        hop1_docs = self.retrieve_k(claim).passages
        hop1_extraction = self.extract_hop1(claim=claim, passages=hop1_docs)
        key_entities_1 = hop1_extraction.key_entities
        key_facts_1 = hop1_extraction.key_facts
        connections_1 = hop1_extraction.connections

        # HOP 2: Generate query using structured info and extract from new passages
        hop2_query = self.create_query_hop2(
            claim=claim,
            key_entities_1=key_entities_1,
            key_facts_1=key_facts_1,
            connections_1=connections_1,
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_extraction = self.extract_hop2(
            claim=claim,
            context=f"Entities: {key_entities_1}, Facts: {key_facts_1}, Connections: {connections_1}",
            passages=hop2_docs,
        )
        key_entities_2 = hop2_extraction.key_entities
        key_facts_2 = hop2_extraction.key_facts
        connections_2 = hop2_extraction.connections

        # HOP 3: Generate query using all structured info from both hops
        hop3_query = self.create_query_hop3(
            claim=claim,
            key_entities_1=key_entities_1,
            key_facts_1=key_facts_1,
            connections_1=connections_1,
            key_entities_2=key_entities_2,
            key_facts_2=key_facts_2,
            connections_2=connections_2,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


