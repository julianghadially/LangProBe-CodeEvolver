import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapAnalysis(dspy.Signature):
    """Analyze a claim to identify key entities, relationships, and information gaps that need to be filled through multi-hop retrieval."""

    claim: str = dspy.InputField()
    key_entities: str = dspy.OutputField(desc="description of the main entities, people, places, or concepts mentioned in the claim")
    information_needed: str = dspy.OutputField(desc="description of what specific facts, relationships, or context need to be verified to assess the claim")
    search_strategy: str = dspy.OutputField(desc="description of how to structure the multi-hop retrieval process, including what to search first and what follow-up information to seek")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.gap_analysis = dspy.ChainOfThought(GapAnalysis)
        self.create_query_hop1 = dspy.ChainOfThought("claim,key_entities,information_needed,search_strategy->query")
        self.create_query_hop2 = dspy.ChainOfThought("claim,key_entities,information_needed,search_strategy,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,key_entities,information_needed,search_strategy,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # Gap Analysis: Identify key entities and information needs before retrieval
        analysis = self.gap_analysis(claim=claim)
        key_entities = analysis.key_entities
        information_needed = analysis.information_needed
        search_strategy = analysis.search_strategy

        # HOP 1: Use gap analysis to generate targeted first query
        hop1_query = self.create_query_hop1(
            claim=claim,
            key_entities=key_entities,
            information_needed=information_needed,
            search_strategy=search_strategy
        ).query
        hop1_docs = self.retrieve_k(hop1_query).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Use gap analysis context for second query
        hop2_query = self.create_query_hop2(
            claim=claim,
            key_entities=key_entities,
            information_needed=information_needed,
            search_strategy=search_strategy,
            summary_1=summary_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Use gap analysis context for third query
        hop3_query = self.create_query_hop3(
            claim=claim,
            key_entities=key_entities,
            information_needed=information_needed,
            search_strategy=search_strategy,
            summary_1=summary_1,
            summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
