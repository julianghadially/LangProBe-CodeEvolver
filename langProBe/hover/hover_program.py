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
        self.create_query_hop2 = dspy.ChainOfThought("claim,key_facts_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,key_facts_1,key_facts_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.extract_facts1 = dspy.ChainOfThought("claim,passages->key_facts")
        self.extract_facts2 = dspy.ChainOfThought("claim,key_facts_1,passages->key_facts")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        key_facts_1 = self.extract_facts1(
            claim=claim, passages=hop1_docs
        ).key_facts  # Extract entities, relationships, and missing information

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, key_facts_1=key_facts_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        key_facts_2 = self.extract_facts2(
            claim=claim, key_facts_1=key_facts_1, passages=hop2_docs
        ).key_facts

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, key_facts_1=key_facts_1, key_facts_2=key_facts_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
