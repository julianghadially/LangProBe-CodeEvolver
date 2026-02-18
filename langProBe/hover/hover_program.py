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
        self.extract_entities = dspy.ChainOfThought("claim,passages->entities")
        self.create_query_hop2 = dspy.ChainOfThought("claim,entities_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,entities_1,entities_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Initial retrieval based on claim
        hop1_docs = self.retrieve_k(claim).passages
        entities_1 = self.extract_entities(
            claim=claim, passages=hop1_docs
        ).entities  # Extract key entities and relationships from first hop

        # HOP 2: Query targeting next entity in reasoning chain
        hop2_query = self.create_query_hop2(claim=claim, entities_1=entities_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        entities_2 = self.extract_entities(
            claim=claim, passages=hop2_docs
        ).entities  # Extract entities and relationships from second hop

        # HOP 3: Query focusing on final entities needed for verification
        hop3_query = self.create_query_hop3(
            claim=claim, entities_1=entities_1, entities_2=entities_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
