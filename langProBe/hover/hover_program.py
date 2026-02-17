import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.extract_entities_hop1 = dspy.ChainOfThought("claim,passages->entities,relationships")
        self.extract_entities_hop2 = dspy.ChainOfThought("claim,entities_1,relationships_1,passages->entities,relationships")
        self.create_query_hop2 = dspy.ChainOfThought("claim,entities,relationships->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,entities_1,relationships_1,entities_2,relationships_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Retrieve initial documents and extract entities
        hop1_docs = self.retrieve_k(claim).passages
        hop1_result = self.extract_entities_hop1(
            claim=claim, passages=hop1_docs
        )  # Extract key entities and relationships from top k docs
        entities_1 = hop1_result.entities
        relationships_1 = hop1_result.relationships

        # HOP 2: Generate entity-focused query to find bridging documents
        hop2_query = self.create_query_hop2(
            claim=claim, entities=entities_1, relationships=relationships_1
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_result = self.extract_entities_hop2(
            claim=claim,
            entities_1=entities_1,
            relationships_1=relationships_1,
            passages=hop2_docs
        )
        entities_2 = hop2_result.entities
        relationships_2 = hop2_result.relationships

        # HOP 3: Generate final query based on accumulated entity knowledge
        hop3_query = self.create_query_hop3(
            claim=claim,
            entities_1=entities_1,
            relationships_1=relationships_1,
            entities_2=entities_2,
            relationships_2=relationships_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
