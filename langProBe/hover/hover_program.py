import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimDecomposition(dspy.Signature):
    """Decompose a claim into 2 focused sub-queries that target different entities or concepts within the claim.
    Each sub-query should focus on a distinct entity, concept, or aspect of the claim to enable parallel retrieval
    of documents about all entities mentioned rather than following a single sequential path."""

    claim: str = dspy.InputField(desc="The claim to decompose into sub-queries")
    sub_queries: list[str] = dspy.OutputField(desc="2 focused sub-queries, each targeting a different entity or concept in the claim")


class GapAnalysis(dspy.Signature):
    """Analyze the initial retrieved documents to identify missing entities or concepts not adequately covered.
    Use chain-of-thought reasoning to determine what critical information gaps exist that would help verify the claim.
    Output 1 targeted follow-up query to address the most important gap."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    initial_documents: str = dspy.InputField(desc="Summary of initially retrieved document titles")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning about what entities or concepts are missing from initial retrieval")
    follow_up_query: str = dspy.OutputField(desc="1 targeted follow-up query to retrieve documents addressing the identified gap")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the original claim on a 1-10 scale with reasoning.
    Provide chain-of-thought reasoning explaining why the document is or isn't relevant to verifying the claim."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    document: str = dspy.InputField(desc="The document to score for relevance")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning about the document's relevance to the claim")
    score: int = dspy.OutputField(desc="Relevance score from 1-10, where 10 is highly relevant and 1 is not relevant")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using iterative retrieval with gap analysis.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Phase 1: Initial retrieval - k=50 per sub-query
        self.initial_k = 50
        self.decompose = dspy.ChainOfThought(ClaimDecomposition)
        self.retrieve_initial = dspy.Retrieve(k=self.initial_k)

        # Phase 2: Gap analysis and follow-up retrieval
        self.gap_analysis = dspy.ChainOfThought(GapAnalysis)
        self.retrieve_followup = dspy.Retrieve(k=50)

        # Phase 3: Reranking
        self.score_relevance = dspy.ChainOfThought(RelevanceScorer)

    def forward(self, claim):
        # PHASE 1: Initial retrieval - decompose into 2 sub-queries, retrieve k=50 each (100 total)
        decomposition = self.decompose(claim=claim)
        sub_queries = decomposition.sub_queries

        # Ensure we have exactly 2 sub-queries
        if not isinstance(sub_queries, list):
            sub_queries = [sub_queries]
        # Take first 2 sub-queries
        sub_queries = sub_queries[:2]
        if len(sub_queries) < 2:
            # Fallback: if decomposition produces less than 2, add the original claim
            sub_queries.append(claim)
        # Ensure exactly 2 queries
        sub_queries = sub_queries[:2]

        # Retrieve k=50 documents per sub-query (100 total)
        all_initial_docs = []
        for sub_query in sub_queries:
            docs = self.retrieve_initial(sub_query).passages
            all_initial_docs.extend(docs)

        # PHASE 2: Gap analysis - identify missing entities/concepts
        # Create summary of initial document titles for gap analysis
        initial_titles = []
        for doc in all_initial_docs:
            title = doc.split(" | ")[0] if " | " in doc else doc
            initial_titles.append(title)

        # Limit to first 50 titles to avoid overly long context
        initial_titles_summary = "\n".join(initial_titles[:50])

        # Perform gap analysis to generate 1 follow-up query
        gap_result = self.gap_analysis(
            claim=claim,
            initial_documents=initial_titles_summary
        )
        follow_up_query = gap_result.follow_up_query

        # Retrieve k=50 documents for follow-up query (50 more docs, 150 total)
        followup_docs = self.retrieve_followup(follow_up_query).passages

        # Combine all documents: initial (100) + follow-up (50) = 150 total
        all_docs = all_initial_docs + followup_docs

        # PHASE 3: Two-stage reranking - score and deduplicate to get top 21
        # Score each document for relevance with chain-of-thought reasoning
        scored_docs = []
        for doc in all_docs:
            try:
                result = self.score_relevance(claim=claim, document=doc)
                score = result.score
                # Ensure score is an integer between 1-10
                if isinstance(score, str):
                    score = int(score)
                score = max(1, min(10, score))
                scored_docs.append((doc, score))
            except (ValueError, AttributeError):
                # If scoring fails, assign a default middle score
                scored_docs.append((doc, 5))

        # Deduplicate by document title and select top 21 unique documents by score
        # Document format is "title | content", extract title for deduplication
        seen_titles = set()
        unique_docs = []

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        for doc, score in scored_docs:
            # Extract title (before the " | " separator)
            title = doc.split(" | ")[0] if " | " in doc else doc
            normalized_title = title.lower().strip()

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

                # Stop once we have 21 unique documents
                if len(unique_docs) >= 21:
                    break

        return dspy.Prediction(retrieved_docs=unique_docs)
