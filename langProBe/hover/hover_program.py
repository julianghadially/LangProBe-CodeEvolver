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


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi-hop retrieval with Chain-of-Thought verification.

    EVALUATION
    - This system retrieves documents and performs verification
    - Returns a label (0 or 1) indicating if claim is supported
    """

    def __init__(self):
        super().__init__()
        # Use existing retrieval pipeline
        self.retrieval = HoverMultiHop()

        # Add Chain-of-Thought verifier
        from .hover_cot_verifier import ChainOfThoughtVerifier

        self.verifier = ChainOfThoughtVerifier()

    def forward(self, claim):
        # STAGE 1: Multi-hop retrieval (existing)
        retrieval_pred = self.retrieval(claim=claim)
        retrieved_docs = retrieval_pred.retrieved_docs

        # STAGE 2: Chain-of-Thought verification (NEW)
        verification_pred = self.verifier(
            claim=claim, retrieved_docs=retrieved_docs
        )

        # Return both retrieval results and verification decision
        return dspy.Prediction(
            retrieved_docs=retrieved_docs,
            label=verification_pred.label,
            verification_decision=verification_pred.verification_decision,
            facts=verification_pred.facts,
            reasoning_steps=verification_pred.reasoning_steps,
            comparisons=verification_pred.comparisons,
        )
