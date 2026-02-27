import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GenerateEntityQuery(dspy.Signature):
    """Generate an entity-focused search query from the claim that targets key entities, people, places, or organizations mentioned."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="an entity-focused search query")


class GenerateRelationshipQuery(dspy.Signature):
    """Generate a relationship-focused search query from the claim that targets connections, events, or relationships between entities."""
    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a relationship-focused search query")


class RerankForCoverage(dspy.Signature):
    """Given a claim and retrieved documents, identify which documents provide the best coverage of entities, relationships, and facts mentioned in the claim. Select documents that collectively cover different aspects of the claim to support multi-hop reasoning chains."""
    claim: str = dspy.InputField()
    documents: str = dspy.InputField(desc="all retrieved documents, numbered")
    selected_indices: str = dspy.OutputField(desc="comma-separated list of exactly 21 document indices (e.g., '0,3,5,7,...') that provide the best coverage")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.retrieve_25 = dspy.Retrieve(k=25)
        self.entity_query_gen = dspy.Predict(GenerateEntityQuery)
        self.relationship_query_gen = dspy.Predict(GenerateRelationshipQuery)
        self.reranker = dspy.Predict(RerankForCoverage)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Initial retrieval with diverse queries
            # Generate entity-focused query
            entity_query = self.entity_query_gen(claim=claim).query
            entity_docs = self.retrieve_25(entity_query).passages

            # Generate relationship-focused query
            relationship_query = self.relationship_query_gen(claim=claim).query
            relationship_docs = self.retrieve_25(relationship_query).passages

            # Combine all 50 documents
            all_docs = entity_docs + relationship_docs

            # Stage 2: Coverage-based reranking
            # Format documents with indices for the reranker
            doc_list_str = "\n".join([f"{i}. {doc}" for i, doc in enumerate(all_docs)])

            # Get reranked indices
            rerank_result = self.reranker(claim=claim, documents=doc_list_str)
            selected_indices_str = rerank_result.selected_indices

            # Parse the indices and select top 21 documents
            try:
                # Parse comma-separated indices
                selected_indices = [int(idx.strip()) for idx in selected_indices_str.split(',')]
                # Filter valid indices and limit to 21
                selected_indices = [idx for idx in selected_indices if 0 <= idx < len(all_docs)][:21]

                # Select the documents based on indices
                reranked_docs = [all_docs[idx] for idx in selected_indices]

                # If we got fewer than 21 documents, pad with remaining docs
                if len(reranked_docs) < 21:
                    remaining_docs = [doc for i, doc in enumerate(all_docs) if i not in selected_indices]
                    reranked_docs.extend(remaining_docs[:21 - len(reranked_docs)])
            except (ValueError, IndexError):
                # Fallback: if parsing fails, just take first 21 documents
                reranked_docs = all_docs[:21]

            return dspy.Prediction(retrieved_docs=reranked_docs)
