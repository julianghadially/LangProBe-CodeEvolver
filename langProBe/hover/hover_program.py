import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapAnalysis(dspy.Signature):
    """Analyze what information is still missing to fully verify or assess the claim based on documents retrieved so far."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    retrieved_so_far: str = dspy.InputField(desc="Summary of information retrieved in previous hops")
    missing_information: str = dspy.OutputField(desc="Description of what critical information is still needed to verify the claim")


class AdaptiveQueryGenerator(dspy.Signature):
    """Generate a targeted search query based on identified information gaps."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    gap_analysis: str = dspy.InputField(desc="Analysis of what information is missing")
    previous_summaries: str = dspy.InputField(desc="Summaries from previous retrieval hops")
    query: str = dspy.OutputField(desc="A specific search query targeting the information gaps")


class DocumentScorer(dspy.Signature):
    """Score a document's relevance to verifying the given claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    document: str = dspy.InputField(desc="The document to score")
    relevance_score: float = dspy.OutputField(desc="Relevance score from 0.0 to 10.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the score")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 20  # Retrieve 20 documents per hop
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Gap analysis and adaptive query generation
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysis)
        self.adaptive_query_gen = dspy.ChainOfThought(AdaptiveQueryGenerator)

        # Document scoring
        self.document_scorer = dspy.ChainOfThought(DocumentScorer)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        all_docs = []

        # HOP 1: Initial retrieval with adaptive query
        gap_analysis_1 = "No information retrieved yet. Need to find relevant documents about the claim."
        previous_summaries_1 = "None"
        hop1_query = self.adaptive_query_gen(
            claim=claim,
            gap_analysis=gap_analysis_1,
            previous_summaries=previous_summaries_1
        ).query

        hop1_docs = self.retrieve_k(hop1_query).passages
        all_docs.extend(hop1_docs)

        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs[:7]  # Summarize top 7 for context
        ).summary

        # HOP 2: Gap analysis + adaptive retrieval
        gap_analysis_2 = self.gap_analyzer(
            claim=claim,
            retrieved_so_far=summary_1
        ).missing_information

        hop2_query = self.adaptive_query_gen(
            claim=claim,
            gap_analysis=gap_analysis_2,
            previous_summaries=summary_1
        ).query

        hop2_docs = self.retrieve_k(hop2_query).passages
        all_docs.extend(hop2_docs)

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs[:7]
        ).summary

        # HOP 3: Final gap analysis + adaptive retrieval
        gap_analysis_3 = self.gap_analyzer(
            claim=claim,
            retrieved_so_far=f"{summary_1} {summary_2}"
        ).missing_information

        hop3_query = self.adaptive_query_gen(
            claim=claim,
            gap_analysis=gap_analysis_3,
            previous_summaries=f"{summary_1} {summary_2}"
        ).query

        hop3_docs = self.retrieve_k(hop3_query).passages
        all_docs.extend(hop3_docs)

        # Score all 60 documents
        scored_docs = []
        for doc in all_docs:
            try:
                result = self.document_scorer(claim=claim, document=doc)
                score = float(result.relevance_score)
                scored_docs.append((doc, score))
            except (ValueError, AttributeError):
                # If scoring fails, assign a default low score
                scored_docs.append((doc, 0.0))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by document title (first part before " | ")
        seen_titles = set()
        unique_docs = []
        for doc, score in scored_docs:
            title = doc.split(" | ")[0]
            # Normalize title for comparison
            normalized_title = title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)
                if len(unique_docs) >= 21:
                    break

        # Ensure we return exactly 21 documents (or less if we don't have enough unique ones)
        final_docs = unique_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
