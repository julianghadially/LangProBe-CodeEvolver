import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimDecomposerSignature(dspy.Signature):
    """Decompose a factual claim into 2-3 independent sub-queries.
    Each sub-query should target a distinct entity, relationship, or verifiable aspect of the claim.
    Sub-queries should be self-contained and complementary, covering different facets needed for verification."""

    claim = dspy.InputField(desc="The factual claim to decompose")
    previous_docs = dspy.InputField(
        desc="Document titles from previous hops (empty list for hop 1). Use this to identify entities/aspects already covered and focus on under-represented entities.",
        default="[]"
    )

    sub_queries = dspy.OutputField(
        desc="List of 2-3 independent search queries as a Python list. Each query should target a distinct entity, relationship, or aspect. For hops after the first, focus on entities/aspects not yet strongly represented in previous_docs."
    )
    rationale = dspy.OutputField(
        desc="Brief explanation of how these sub-queries decompose the claim and what distinct aspect each targets"
    )


class DocumentRelevanceScorerSignature(dspy.Signature):
    """Score how relevant a document is to verifying a factual claim.
    Consider whether the document contains information about entities, relationships, dates, or facts mentioned in the claim."""

    claim = dspy.InputField(desc="The factual claim being verified")
    document = dspy.InputField(desc="Document in 'title | content' format to score")

    relevance_score = dspy.OutputField(
        desc="Relevance score from 0-10, where 0 is completely irrelevant and 10 is highly relevant for verifying the claim. Output only the numeric score."
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of the score, noting which entities/facts from the claim are covered"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval system with query decomposition and fusion retrieval.
    Decomposes claims into 2-3 independent sub-queries per hop, retrieves 35 docs per sub-query,
    merges results with deduplication, and reranks merged pool to top-7 docs per hop.

    EVALUATION
    - Returns exactly 21 documents (7 per hop × 3 hops)
    - Retrieves up to 105 raw docs per hop (35*3 sub-queries) for diverse coverage
    - Aggressive reranking maintains quality through DocumentRelevanceScorer'''

    def __init__(self):
        super().__init__()
        self.k = 7  # Top K docs per hop after reranking
        self.fusion_k = 35  # Retrieve K docs per sub-query for fusion

        # Single retriever instance for fusion retrieval
        self.retrieve_fusion = dspy.Retrieve(k=self.fusion_k)

        # Query decomposition modules (one per hop)
        self.decomposer_hop1 = dspy.ChainOfThought(ClaimDecomposerSignature)
        self.decomposer_hop2 = dspy.ChainOfThought(ClaimDecomposerSignature)
        self.decomposer_hop3 = dspy.ChainOfThought(ClaimDecomposerSignature)

        # Document relevance scorer (shared across all hops)
        self.relevance_scorer = dspy.ChainOfThought(DocumentRelevanceScorerSignature)

    def _extract_titles(self, passages: list[str]) -> list[str]:
        """Extract document titles from passages in 'title | content' format"""
        return [passage.split(" | ")[0] for passage in passages]

    def _fusion_retrieve_and_rerank(self, claim: str, sub_queries: list, previous_titles: list[str]) -> list[str]:
        """
        Perform fusion retrieval and reranking.

        Args:
            claim: The original claim for relevance scoring
            sub_queries: List of decomposed sub-queries (2-3 queries) or string representation
            previous_titles: Already retrieved document titles (for deduplication)

        Returns:
            List of top-7 reranked document passages in 'title | content' format
        """
        # Parse sub_queries if it's a string representation of a list
        if isinstance(sub_queries, str):
            import ast
            try:
                sub_queries = ast.literal_eval(sub_queries)
            except:
                sub_queries = [sub_queries]

        # Ensure we have a list
        if not isinstance(sub_queries, list):
            sub_queries = [str(sub_queries)]

        # Cap at 3 queries to maintain retrieval budget
        sub_queries = sub_queries[:3]

        # Step 1: Retrieve for each sub-query
        all_passages = []
        for sub_query in sub_queries:
            passages = self.retrieve_fusion(sub_query).passages
            all_passages.extend(passages)

        # Step 2: Deduplicate by title (preserves first occurrence)
        seen_titles = set(previous_titles)  # Avoid previously retrieved docs
        unique_passages = []
        for passage in all_passages:
            title = passage.split(" | ")[0]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_passages.append(passage)

        # Step 3: Score each unique document
        scored_docs = []
        for passage in unique_passages:
            score_output = self.relevance_scorer(claim=claim, document=passage)
            try:
                # Parse score as float (handle "8.5" or "8/10" formats)
                score_str = str(score_output.relevance_score).strip()
                if '/' in score_str:
                    score = float(score_str.split('/')[0])
                else:
                    score = float(score_str)
            except (ValueError, AttributeError):
                score = 0.0  # Default to 0 if parsing fails
            scored_docs.append((passage, score))

        # Step 4: Sort by relevance score (descending) and take top K
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k_docs = [doc for doc, score in scored_docs[:self.k]]

        return top_k_docs

    def forward(self, claim):
        all_retrieved_titles = []

        # HOP 1: Initial decomposition and fusion retrieval
        decomp1 = self.decomposer_hop1(claim=claim, previous_docs="[]")
        hop1_docs = self._fusion_retrieve_and_rerank(
            claim=claim,
            sub_queries=decomp1.sub_queries,
            previous_titles=[]
        )
        hop1_titles = self._extract_titles(hop1_docs)
        all_retrieved_titles.extend(hop1_titles)

        # HOP 2: Context-aware decomposition using hop1 results
        decomp2 = self.decomposer_hop2(
            claim=claim,
            previous_docs=str(hop1_titles)  # Pass titles as context
        )
        hop2_docs = self._fusion_retrieve_and_rerank(
            claim=claim,
            sub_queries=decomp2.sub_queries,
            previous_titles=all_retrieved_titles
        )
        hop2_titles = self._extract_titles(hop2_docs)
        all_retrieved_titles.extend(hop2_titles)

        # HOP 3: Final gap-filling decomposition
        decomp3 = self.decomposer_hop3(
            claim=claim,
            previous_docs=str(all_retrieved_titles)  # All previous titles
        )
        hop3_docs = self._fusion_retrieve_and_rerank(
            claim=claim,
            sub_queries=decomp3.sub_queries,
            previous_titles=all_retrieved_titles
        )

        # Return all 21 documents (maintains evaluation contract)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
