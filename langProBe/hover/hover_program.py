import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityAndGapAnalyzer(dspy.Signature):
    """Analyze the claim to extract multiple entity chains and identify information gaps that need to be verified.

    Entity chains are sequences of related entities mentioned in the claim (e.g., person -> organization -> location).
    Information gaps are missing pieces of information needed to verify the claim.
    Generate 2-3 distinct search queries targeting different entity chains or information gaps."""

    claim: str = dspy.InputField(desc="The claim to analyze for entities and information gaps")
    entity_chains: str = dspy.OutputField(desc="2-3 distinct entity chains or key topics extracted from the claim, separated by newlines")
    queries: list[str] = dspy.OutputField(desc="2-3 parallel search queries targeting different entity chains or information gaps (must be 2-3 queries)")


class DocumentRelevanceScorer(dspy.Signature):
    """Score each document's relevance to the claim and identified entity chains.

    Consider:
    - How well the document addresses the claim
    - Coverage of mentioned entity chains
    - Factual information that helps verify the claim

    Return a relevance score from 0-100 for each document."""

    claim: str = dspy.InputField(desc="The original claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    document: str = dspy.InputField(desc="A retrieved document to score")
    relevance_score: int = dspy.OutputField(desc="Relevance score from 0-100 indicating how relevant this document is to verifying the claim")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Parallel retrieval with larger k per query
        self.k_per_query = 23  # Retrieve 23 documents per query (middle of 21-25 range)
        self.max_final_docs = 21  # Final output constraint

        # Entity and gap analysis module
        self.entity_gap_analyzer = dspy.ChainOfThought(EntityAndGapAnalyzer)

        # Retrieval module
        self.retrieve_k = dspy.Retrieve(k=self.k_per_query)

        # Document relevance scorer
        self.doc_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)

    def forward(self, claim):
        # Step 1: Analyze claim to extract entity chains and generate parallel queries
        analysis = self.entity_gap_analyzer(claim=claim)
        entity_chains = analysis.entity_chains
        queries = analysis.queries

        # Ensure we have 2-3 queries (constraint: max 3 queries per claim)
        if len(queries) < 2:
            # If only 1 query generated, duplicate with slight variation
            queries = [queries[0], claim][:3]
        elif len(queries) > 3:
            # If more than 3, take top 3
            queries = queries[:3]

        # Step 2: Parallel retrieval - retrieve k documents for each query
        all_retrieved_docs = []
        for query in queries:
            docs = self.retrieve_k(query).passages
            all_retrieved_docs.extend(docs)

        # Deduplicate documents while preserving order
        seen = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)

        # Step 3: Score-based reranking
        # If we have fewer than max_final_docs, return all
        if len(unique_docs) <= self.max_final_docs:
            return dspy.Prediction(retrieved_docs=unique_docs)

        # Score each document
        scored_docs = []
        for doc in unique_docs:
            try:
                score_result = self.doc_scorer(
                    claim=claim,
                    entity_chains=entity_chains,
                    document=doc
                )
                score = score_result.relevance_score
                # Ensure score is an integer
                if isinstance(score, str):
                    # Extract first number from string if needed
                    import re
                    match = re.search(r'\d+', score)
                    score = int(match.group()) if match else 50
                scored_docs.append((doc, int(score)))
            except Exception:
                # If scoring fails, assign middle score
                scored_docs.append((doc, 50))

        # Sort by score descending and take top max_final_docs
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:self.max_final_docs]]

        return dspy.Prediction(retrieved_docs=top_docs)
