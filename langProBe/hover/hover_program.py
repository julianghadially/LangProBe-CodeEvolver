import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimDecomposition(dspy.Signature):
    """Decompose a claim into 2-3 focused sub-queries that target different entities or concepts within the claim.
    Each sub-query should focus on a distinct entity, concept, or aspect of the claim to enable parallel retrieval
    of documents about all entities mentioned rather than following a single sequential path."""

    claim: str = dspy.InputField(desc="The claim to decompose into sub-queries")
    sub_queries: list[str] = dspy.OutputField(desc="2-3 focused sub-queries, each targeting a different entity or concept in the claim")


class QuickRelevanceFilter(dspy.Signature):
    """Quickly determine if a document is potentially relevant to the claim with a binary yes/no decision.
    This is a fast filtering step to reduce the number of documents before detailed scoring."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    document: str = dspy.InputField(desc="The document to evaluate for potential relevance")
    is_relevant: str = dspy.OutputField(desc="Answer 'yes' if the document is potentially relevant to the claim, 'no' otherwise")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the original claim on a 1-10 scale with reasoning.
    Provide chain-of-thought reasoning explaining why the document is or isn't relevant to verifying the claim."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    document: str = dspy.InputField(desc="The document to score for relevance")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning about the document's relevance to the claim")
    score: int = dspy.OutputField(desc="Relevance score from 1-10, where 10 is highly relevant and 1 is not relevant")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using parallel multi-entity query decomposition.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 100  # Retrieve 100 documents per sub-query (up to 300 total)
        self.decompose = dspy.ChainOfThought(ClaimDecomposition)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.quick_filter = dspy.Predict(QuickRelevanceFilter)  # First stage: fast binary filtering
        self.score_relevance = dspy.ChainOfThought(RelevanceScorer)  # Second stage: detailed scoring

    def forward(self, claim):
        # Step 1: Decompose claim into 2-3 sub-queries targeting different entities
        decomposition = self.decompose(claim=claim)
        sub_queries = decomposition.sub_queries

        # Ensure we have 2-3 sub-queries (handle edge cases)
        if not isinstance(sub_queries, list):
            sub_queries = [sub_queries]
        sub_queries = sub_queries[:3]  # Limit to max 3 sub-queries
        if len(sub_queries) < 2:
            # Fallback: if decomposition produces less than 2, add the original claim
            sub_queries.append(claim)

        # Step 2: Retrieve k=100 documents per sub-query in parallel (up to 300 total)
        all_docs = []
        for sub_query in sub_queries:
            docs = self.retrieve_k(sub_query).passages
            all_docs.extend(docs)

        # Step 3: First-stage reranking - Quick binary filtering to reduce ~300 docs to ~60
        filtered_docs = []
        for doc in all_docs:
            try:
                result = self.quick_filter(claim=claim, document=doc)
                is_relevant = result.is_relevant.lower().strip()
                # Accept if the answer contains 'yes'
                if 'yes' in is_relevant:
                    filtered_docs.append(doc)
            except (AttributeError, Exception):
                # If filtering fails, include the document to be safe
                filtered_docs.append(doc)

        # Limit to top ~60 documents after filtering (if we get more than expected)
        # We'll keep all filtered docs but cap at a reasonable number for the second stage
        if len(filtered_docs) > 60:
            filtered_docs = filtered_docs[:60]

        # Step 4: Second-stage reranking - Detailed scoring with chain-of-thought on filtered docs
        scored_docs = []
        for doc in filtered_docs:
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

        # Step 5: Deduplicate by document title and select top 21 unique documents by score
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
