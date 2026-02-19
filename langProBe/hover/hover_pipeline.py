import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class GapAnalysisSignature(dspy.Signature):
    """Analyze the claim and previously retrieved documents to identify specific missing information entities (people, places, events, dates, organizations) that are needed to verify the claim but are not covered in the current documents."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    retrieved_docs: str = dspy.InputField(desc="Previously retrieved documents as concatenated text")
    missing_info: str = dspy.OutputField(desc="Specific entities, facts, or information gaps (people, places, events, dates) that are mentioned in the claim but not adequately covered in the retrieved documents")


class GapAwareQueryGenerator(dspy.Signature):
    """Generate a targeted search query to fill specific information gaps identified in the gap analysis."""

    claim: str = dspy.InputField(desc="The original claim to be verified")
    missing_info: str = dspy.InputField(desc="Specific missing information entities and facts identified by gap analysis")
    query: str = dspy.OutputField(desc="A focused search query targeting the missing information to fill the identified gaps")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to verifying the given claim on a scale from 0-10, where 10 is highly relevant and directly helps verify the claim, and 0 is completely irrelevant."""

    claim: str = dspy.InputField(desc="The claim to be verified")
    document: str = dspy.InputField(desc="A retrieved document to score")
    score: int = dspy.OutputField(desc="Relevance score from 0-10, where 10 means highly relevant to verifying the claim")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with gap-aware retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize gap-aware components
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysisSignature)
        self.gap_query_generator = dspy.ChainOfThought(GapAwareQueryGenerator)
        self.relevance_scorer = dspy.Predict(RelevanceScorer)

        # Retrieve more documents for reranking
        self.k_retrieve = 30
        self.k_final = 7  # Final documents per hop after reranking
        self.retrieve_k = dspy.Retrieve(k=self.k_retrieve)

        # Query generators and summarizers
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _score_and_rerank_docs(self, claim, docs, k=7):
        """Score documents by relevance and return top k."""
        if len(docs) <= k:
            return docs

        scored_docs = []
        for doc in docs:
            try:
                score_result = self.relevance_scorer(claim=claim, document=doc)
                score = int(score_result.score) if hasattr(score_result, 'score') else 5
                # Clamp score to valid range
                score = max(0, min(10, score))
            except:
                # Default to middle score if scoring fails
                score = 5
            scored_docs.append((score, doc))

        # Sort by score descending and return top k documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # HOP 1: Initial retrieval with claim
            hop1_docs_raw = self.retrieve_k(claim).passages
            hop1_docs = self._score_and_rerank_docs(claim, hop1_docs_raw, k=self.k_final)

            # Summarize hop 1 results
            summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

            # GAP ANALYSIS AFTER HOP 1
            # Concatenate hop1 docs for gap analysis
            hop1_docs_text = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(hop1_docs)])
            gap_analysis_1 = self.gap_analyzer(
                claim=claim,
                retrieved_docs=hop1_docs_text
            )
            missing_info_1 = gap_analysis_1.missing_info

            # HOP 2: Generate gap-aware query targeting missing information
            hop2_query = self.gap_query_generator(
                claim=claim,
                missing_info=missing_info_1
            ).query

            hop2_docs_raw = self.retrieve_k(hop2_query).passages
            hop2_docs = self._score_and_rerank_docs(claim, hop2_docs_raw, k=self.k_final)

            # Summarize hop 2 results
            summary_2 = self.summarize2(
                claim=claim,
                context=summary_1,
                passages=hop2_docs
            ).summary

            # GAP ANALYSIS AFTER HOP 2
            # Concatenate all docs retrieved so far
            all_docs_so_far = hop1_docs + hop2_docs
            all_docs_text = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(all_docs_so_far)])
            gap_analysis_2 = self.gap_analyzer(
                claim=claim,
                retrieved_docs=all_docs_text
            )
            missing_info_2 = gap_analysis_2.missing_info

            # HOP 3: Generate final gap-aware query for remaining gaps
            hop3_query = self.gap_query_generator(
                claim=claim,
                missing_info=missing_info_2
            ).query

            hop3_docs_raw = self.retrieve_k(hop3_query).passages
            hop3_docs = self._score_and_rerank_docs(claim, hop3_docs_raw, k=self.k_final)

            # Combine all documents (3 hops × 7 docs = 21 total)
            all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs

            return dspy.Prediction(retrieved_docs=all_retrieved_docs)
