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


class ListwiseDocumentRanker(dspy.Signature):
    """Rank all retrieved documents by relevance to the claim using comparative judgment across all documents simultaneously.

    This listwise ranking approach allows you to make direct comparisons between documents to identify the most relevant ones.
    Prioritize documents that:
    - Directly address and help verify the claim
    - Cover different entity chains mentioned in the claim (ensure multi-hop coverage across entity chains)
    - Provide complementary information that together supports comprehensive fact verification
    - Contain factual evidence rather than tangential information

    Return a ranked list of document indices (0 to len(documents)-1) ordered from most to least relevant."""

    claim: str = dspy.InputField(desc="The original claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim that need coverage")
    documents: list[str] = dspy.InputField(desc="All retrieved documents to rank (indexed 0 to len-1)")
    ranked_indices: list[int] = dspy.OutputField(desc="Ranked list of document indices from most to least relevant (e.g., [5, 2, 8, 0, ...])")


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

        # Listwise document ranker (replaces individual scoring)
        self.listwise_ranker = dspy.Predict(ListwiseDocumentRanker)

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

        # Step 3: Listwise reranking
        # If we have fewer than max_final_docs, return all
        if len(unique_docs) <= self.max_final_docs:
            return dspy.Prediction(retrieved_docs=unique_docs)

        # Use listwise ranker to rank all documents in a single pass
        try:
            ranking_result = self.listwise_ranker(
                claim=claim,
                entity_chains=entity_chains,
                documents=unique_docs
            )
            ranked_indices = ranking_result.ranked_indices

            # Validate and sanitize indices
            valid_indices = []
            for idx in ranked_indices:
                try:
                    # Convert to int if string
                    if isinstance(idx, str):
                        idx = int(idx)
                    # Check if index is within valid range
                    if 0 <= idx < len(unique_docs):
                        # Avoid duplicates
                        if idx not in valid_indices:
                            valid_indices.append(idx)
                except (ValueError, TypeError):
                    continue

            # If we got valid indices, use them to reorder documents
            if valid_indices:
                reranked_docs = [unique_docs[i] for i in valid_indices]
                # Take top max_final_docs
                top_docs = reranked_docs[:self.max_final_docs]
            else:
                # Fallback: if ranking failed, return first max_final_docs
                top_docs = unique_docs[:self.max_final_docs]

        except Exception:
            # Fallback: if ranking fails entirely, return first max_final_docs
            top_docs = unique_docs[:self.max_final_docs]

        return dspy.Prediction(retrieved_docs=top_docs)
