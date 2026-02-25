import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimDecomposition(dspy.Signature):
    """Decompose a claim into 2-3 focused sub-queries that target different entities or concepts within the claim.
    Each sub-query should focus on a distinct entity, concept, or aspect of the claim to enable parallel retrieval
    of documents about all entities mentioned rather than following a single sequential path."""

    claim: str = dspy.InputField(desc="The claim to decompose into sub-queries")
    sub_queries: list[str] = dspy.OutputField(desc="2-3 focused sub-queries, each targeting a different entity or concept in the claim")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the original claim on a 1-10 scale with reasoning.
    Provide chain-of-thought reasoning explaining why the document is or isn't relevant to verifying the claim."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    document: str = dspy.InputField(desc="The document to score for relevance")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning about the document's relevance to the claim")
    score: int = dspy.OutputField(desc="Relevance score from 1-10, where 10 is highly relevant and 1 is not relevant")


class MissingEntityDetector(dspy.Signature):
    """Analyze a claim and the titles of retrieved documents to identify 1-2 specific entities or facts from the claim
    that are poorly covered by the current documents. Focus on key entities, dates, locations, or relationships
    mentioned in the claim that don't appear to have dedicated documents."""

    claim: str = dspy.InputField(desc="The claim being fact-checked")
    retrieved_titles: str = dspy.InputField(desc="List of document titles retrieved so far")
    missing_elements: list[str] = dspy.OutputField(desc="1-2 specific entities or facts from the claim that are poorly covered by current documents")


class TargetedQueryGenerator(dspy.Signature):
    """Generate 1 highly specific query to find documents about missing entities or facts from a claim.
    The query should be focused and precise to retrieve documents that fill gaps in the current document pool."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    missing_elements: str = dspy.InputField(desc="Specific entities or facts that are poorly covered")
    targeted_query: str = dspy.OutputField(desc="A highly specific query to find documents about the missing elements")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using parallel multi-entity query decomposition.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25  # Retrieve 25 documents per sub-query
        self.decompose = dspy.ChainOfThought(ClaimDecomposition)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.score_relevance = dspy.ChainOfThought(RelevanceScorer)
        self.detect_gaps = dspy.ChainOfThought(MissingEntityDetector)
        self.generate_targeted_query = dspy.ChainOfThought(TargetedQueryGenerator)
        self.retrieve_targeted = dspy.Retrieve(k=25)  # Gap-filling retrieval

    def forward(self, claim):
        # Step 1: Decompose claim into 2-3 sub-queries targeting different entities
        decomposition = self.decompose(claim=claim)
        sub_queries = decomposition.sub_queries

        # Ensure we have 2-3 sub-queries (handle edge cases)
        if not isinstance(sub_queries, list):
            sub_queries = [sub_queries]
        # Limit to 2 sub-queries to leave room for 1 targeted query (total 3 queries)
        sub_queries = sub_queries[:2]
        if len(sub_queries) < 2:
            # Fallback: if decomposition produces less than 2, add the original claim
            sub_queries.append(claim)

        # Step 2: Retrieve k=25 documents per sub-query in parallel (up to 50 total)
        all_docs = []
        for sub_query in sub_queries:
            docs = self.retrieve_k(sub_query).passages
            all_docs.extend(docs)

        # Step 2.5: Gap detection and targeted retrieval
        # Extract document titles for gap analysis
        doc_titles = []
        for doc in all_docs:
            title = doc.split(" | ")[0] if " | " in doc else doc
            doc_titles.append(title)

        retrieved_titles_str = "\n".join(doc_titles)

        # Detect gaps in coverage
        try:
            gap_result = self.detect_gaps(claim=claim, retrieved_titles=retrieved_titles_str)
            missing_elements = gap_result.missing_elements

            # Ensure missing_elements is a list
            if not isinstance(missing_elements, list):
                missing_elements = [missing_elements]

            # If gaps are detected, perform targeted retrieval
            if missing_elements and len(missing_elements) > 0:
                # Filter out empty or meaningless elements
                missing_elements = [e for e in missing_elements if e and isinstance(e, str) and len(e.strip()) > 0]

                if missing_elements:
                    # Limit to 1-2 missing elements
                    missing_elements = missing_elements[:2]
                    missing_elements_str = ", ".join(missing_elements)

                    # Generate a targeted query to fill the gaps
                    targeted_query_result = self.generate_targeted_query(
                        claim=claim,
                        missing_elements=missing_elements_str
                    )
                    targeted_query = targeted_query_result.targeted_query

                    # Perform targeted retrieval
                    targeted_docs = self.retrieve_targeted(targeted_query).passages
                    all_docs.extend(targeted_docs)
        except (AttributeError, ValueError, Exception):
            # If gap detection fails, continue with existing documents
            pass

        # Step 3: Score each document for relevance with chain-of-thought reasoning
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

        # Step 4: Deduplicate by document title and select top 21 unique documents by score
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
