import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityQueryGenerator(dspy.Signature):
    """Generate three distinct, complementary queries for retrieving documents to verify a claim.
    Query 1 should be a full claim query using the complete claim text.
    Query 2 should focus on the main subject entities mentioned in the claim.
    Query 3 should target connecting concepts, relationships, or context between entities."""

    claim: str = dspy.InputField(desc="The claim to verify")
    full_claim_query: str = dspy.OutputField(desc="Query 1: A comprehensive query using the full claim")
    subject_query: str = dspy.OutputField(desc="Query 2: A query focusing on the main subject entities")
    relation_query: str = dspy.OutputField(desc="Query 3: A query targeting connecting concepts or relationships")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.query_generator = dspy.Predict(EntityQueryGenerator)
        self.retrieve_k50 = dspy.Retrieve(k=50)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Generate three distinct queries
            queries = self.query_generator(claim=claim)

            # Retrieve k=50 documents for each query
            query_list = [
                queries.full_claim_query,
                queries.subject_query,
                queries.relation_query
            ]

            # Store documents with their ranks in each query
            doc_scores = {}  # doc -> RRF score

            for query in query_list:
                passages = self.retrieve_k50(query).passages

                # Apply RRF: score = 1/(60 + rank)
                for rank, passage in enumerate(passages):
                    if passage not in doc_scores:
                        doc_scores[passage] = 0.0
                    doc_scores[passage] += 1.0 / (60 + rank)

            # Sort documents by RRF score (descending) and take top 21
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, score in sorted_docs[:21]]

            return dspy.Prediction(retrieved_docs=retrieved_docs)
