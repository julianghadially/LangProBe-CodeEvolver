import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GapAnalysis(dspy.Signature):
    """Analyze what information is missing after retrieving documents for a claim.
    Identify specific entities, relationships, or facts that are still needed to fully verify the claim."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    retrieved_passages: str = dspy.InputField(desc="The passages retrieved so far")
    current_summary: str = dspy.InputField(desc="Summary of information found so far")
    missing_information: str = dspy.OutputField(desc="Specific entities, relationships, or facts still needed to verify the claim")


class GapAwareQueryGeneration(dspy.Signature):
    """Generate a search query based on the claim, current summary, and identified information gaps."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    summary: str = dspy.InputField(desc="Summary of information found so far")
    missing_information: str = dspy.InputField(desc="Specific information gaps that need to be filled")
    query: str = dspy.OutputField(desc="Search query targeting the missing information")


class DocumentReranker(dspy.Signature):
    """Score and rerank all retrieved documents based on their relevance to the claim.
    Return the indices of the top 21 most relevant documents."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="All retrieved documents with their indices")
    ranked_document_indices: list[int] = dspy.OutputField(desc="Indices of the top 21 most relevant documents, in descending order of relevance")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with gap-aware retrieval and LLM-based reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.k = 20

        # Initialize sub-modules
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize_hop1 = dspy.ChainOfThought("claim, passages -> summary")
        self.gap_analysis_hop1 = dspy.ChainOfThought(GapAnalysis)
        self.gap_aware_query_hop2 = dspy.ChainOfThought(GapAwareQueryGeneration)
        self.summarize_hop2 = dspy.ChainOfThought("claim, context, passages -> summary")
        self.gap_analysis_hop2 = dspy.ChainOfThought(GapAnalysis)
        self.gap_aware_query_hop3 = dspy.ChainOfThought(GapAwareQueryGeneration)
        self.reranker = dspy.ChainOfThought(DocumentReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Initial retrieval
            hop1_docs = self.retrieve_k(claim).passages
            summary_1 = self.summarize_hop1(claim=claim, passages=hop1_docs).summary

            # Gap analysis after hop 1
            gap_analysis_1 = self.gap_analysis_hop1(
                claim=claim,
                retrieved_passages="\n".join(hop1_docs),
                current_summary=summary_1
            ).missing_information

            # HOP 2: Gap-aware retrieval
            hop2_query = self.gap_aware_query_hop2(
                claim=claim,
                summary=summary_1,
                missing_information=gap_analysis_1
            ).query
            hop2_docs = self.retrieve_k(hop2_query).passages
            summary_2 = self.summarize_hop2(
                claim=claim,
                context=summary_1,
                passages=hop2_docs
            ).summary

            # Gap analysis after hop 2
            combined_passages_2 = "\n".join(hop1_docs + hop2_docs)
            gap_analysis_2 = self.gap_analysis_hop2(
                claim=claim,
                retrieved_passages=combined_passages_2,
                current_summary=summary_2
            ).missing_information

            # HOP 3: Gap-aware retrieval
            hop3_query = self.gap_aware_query_hop3(
                claim=claim,
                summary=summary_2,
                missing_information=gap_analysis_2
            ).query
            hop3_docs = self.retrieve_k(hop3_query).passages

            # Combine all 60 documents
            all_docs = hop1_docs + hop2_docs + hop3_docs

            # Rerank all documents
            # Format documents with indices for reranking
            indexed_docs = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(all_docs)])

            # Get top 21 document indices
            ranked_indices = self.reranker(
                claim=claim,
                documents=indexed_docs
            ).ranked_document_indices

            # Select top 21 documents based on ranked indices
            # Ensure we handle cases where the LLM returns fewer or more than 21 indices
            top_indices = ranked_indices[:21] if len(ranked_indices) >= 21 else ranked_indices

            # If we have fewer than 21 indices, pad with remaining documents
            if len(top_indices) < 21:
                remaining_indices = [i for i in range(len(all_docs)) if i not in top_indices]
                top_indices.extend(remaining_indices[:21 - len(top_indices)])

            # Extract the reranked documents
            reranked_docs = [all_docs[i] for i in top_indices if i < len(all_docs)]

            return dspy.Prediction(retrieved_docs=reranked_docs)
