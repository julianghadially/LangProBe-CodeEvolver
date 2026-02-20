import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DecomposeClaimToQueries(dspy.Signature):
    """Decompose a claim into 3 distinct search queries by identifying different entities/concepts
    and generating synonyms or related terms for each. Each query should target different aspects
    of the claim to maximize coverage of relevant documents."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    query1: str = dspy.OutputField(desc="first search query focusing on a specific entity/concept with variations")
    query2: str = dspy.OutputField(desc="second search query focusing on a different entity/concept with synonyms or related terms")
    query3: str = dspy.OutputField(desc="third search query focusing on another distinct entity/concept or aspect")


class RerankDocumentsByRelevance(dspy.Signature):
    """Score and rerank documents by their relevance to the claim using LLM reasoning.
    Analyze which documents contain supporting facts that help verify or refute the claim,
    and return the top 21 most relevant documents."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    documents: list[str] = dspy.InputField(desc="pool of retrieved documents to rerank")
    top_documents: list[str] = dspy.OutputField(desc="top 21 most relevant documents ranked by relevance to the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.decompose = dspy.ChainOfThought(DecomposeClaimToQueries)
        self.retrieve_k50 = dspy.Retrieve(k=50)
        self.rerank = dspy.ChainOfThought(RerankDocumentsByRelevance)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Decompose claim into 3 distinct queries
            decomposed = self.decompose(claim=claim)
            query1 = decomposed.query1
            query2 = decomposed.query2
            query3 = decomposed.query3

            # Stage 2: Retrieve k=50 documents for each query
            docs1 = self.retrieve_k50(query1).passages
            docs2 = self.retrieve_k50(query2).passages
            docs3 = self.retrieve_k50(query3).passages

            # Combine all retrieved documents (up to 150)
            all_docs = docs1 + docs2 + docs3

            # Stage 3: Rerank the combined pool to get top 21 most relevant documents
            reranked = self.rerank(claim=claim, documents=all_docs)
            top_21_docs = reranked.top_documents

            # Ensure we return at most 21 documents
            final_docs = top_21_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
