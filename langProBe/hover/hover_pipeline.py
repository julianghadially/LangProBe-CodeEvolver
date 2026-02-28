import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class KeyPhraseExtraction(dspy.Signature):
    """Extract 2-3 quoted phrases or entity names from the claim that MUST appear verbatim in supporting documents.
    Focus on specific names, locations, dates, or technical terms that can be used for exact matching.
    For example, from 'BB&T Center (Sunrise, Florida)' extract both 'BB&T Center' and 'Sunrise, Florida'."""

    claim: str = dspy.InputField()
    key_phrases: list[str] = dspy.OutputField(desc="2-3 specific quoted phrases or entity names for exact matching")


class BridgingQuery(dspy.Signature):
    """Generate a query to find connecting/bridging documents based on the claim and initial retrieval results.
    The query should help discover intermediate documents that connect entities or concepts."""

    claim: str = dspy.InputField()
    initial_results: str = dspy.InputField(desc="summary of first retrieval results")
    query: str = dspy.OutputField(desc="a bridging query to find connecting documents")


class CoverageReranker(dspy.Signature):
    """Rerank documents by: (1) exact phrase matches from key phrases, (2) entity coverage, (3) claim relevance.
    Return document indices in descending order of relevance (most relevant first)."""

    claim: str = dspy.InputField()
    key_phrases: list[str] = dspy.InputField(desc="key phrases that should appear in documents")
    documents: str = dspy.InputField(desc="all retrieved documents with indices")
    ranked_indices: list[int] = dspy.OutputField(desc="document indices ranked by relevance (0-based)")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using hybrid literal + generated query strategy.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize the signature-based modules
        self.extract_phrases = dspy.Predict(KeyPhraseExtraction)
        self.generate_bridging = dspy.ChainOfThought(BridgingQuery)
        self.rerank_docs = dspy.ChainOfThought(CoverageReranker)

        # Create retrieve modules with different k values
        self.retrieve_15 = dspy.Retrieve(k=15)
        self.retrieve_10 = dspy.Retrieve(k=10)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract key phrases from the claim
            extraction_result = self.extract_phrases(claim=claim)
            key_phrases = extraction_result.key_phrases

            # Ensure we have at least 2 key phrases
            if not key_phrases or len(key_phrases) == 0:
                key_phrases = [claim]  # Fallback to using the claim itself

            # Step 2: Query 1 - Use first extracted phrase with k=15
            query1 = key_phrases[0] if len(key_phrases) > 0 else claim
            hop1_docs = self.retrieve_15(query1).passages

            # Step 3: Query 2 - Generate bridging query with k=10
            # Summarize initial results for context
            initial_summary = " | ".join([doc[:100] for doc in hop1_docs[:3]])  # First 100 chars of top 3 docs
            bridging_result = self.generate_bridging(claim=claim, initial_results=initial_summary)
            bridging_query = bridging_result.query
            hop2_docs = self.retrieve_10(bridging_query).passages

            # Step 4: Query 3 - Use second extracted phrase with k=10
            query3 = key_phrases[1] if len(key_phrases) > 1 else key_phrases[0]
            hop3_docs = self.retrieve_10(query3).passages

            # Step 5: Concatenate all documents (15+10+10=35 total)
            all_docs = hop1_docs + hop2_docs + hop3_docs

            # Step 6: Prepare documents for reranking
            # Format documents with indices for the reranker
            formatted_docs = "\n".join([f"[{i}] {doc[:200]}" for i, doc in enumerate(all_docs)])

            # Rerank documents based on phrase matches, entity coverage, and relevance
            rerank_result = self.rerank_docs(
                claim=claim,
                key_phrases=key_phrases,
                documents=formatted_docs
            )
            ranked_indices = rerank_result.ranked_indices

            # Step 7: Return top 21 documents from reranking
            # Ensure ranked_indices are valid
            valid_indices = [idx for idx in ranked_indices if 0 <= idx < len(all_docs)]

            # If we don't have enough valid indices, append remaining docs in order
            if len(valid_indices) < 21:
                remaining_indices = [i for i in range(len(all_docs)) if i not in valid_indices]
                valid_indices.extend(remaining_indices)

            # Take top 21
            top_21_indices = valid_indices[:21]
            retrieved_docs = [all_docs[idx] for idx in top_21_indices]

            return dspy.Prediction(retrieved_docs=retrieved_docs)
