import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityBasedQuery(dspy.Signature):
    """Generate a search query focused on extracting and querying named entities (people, places, organizations)
    mentioned in the claim and discovered context."""

    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="previously retrieved information")
    query: str = dspy.OutputField(desc="entity-focused search query")


class ContrastiveQuery(dspy.Signature):
    """Generate a search query focused on what's missing, contradictory, or needs verification
    in the current understanding of the claim."""

    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="previously retrieved information")
    query: str = dspy.OutputField(desc="contrastive search query exploring gaps or contradictions")


class RelationalQuery(dspy.Signature):
    """Generate a search query focused on relationships, connections, and interactions
    between concepts mentioned in the claim."""

    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="previously retrieved information")
    query: str = dspy.OutputField(desc="relational search query exploring connections between concepts")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using parallel multi-perspective retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k_per_query = 10  # Retrieve 10 docs per query in parallel hops
        self.k_final = 7  # Select top 7 most novel docs per hop

        # Hop 1 components (unchanged)
        self.retrieve_k = dspy.Retrieve(k=self.k_final)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")

        # Hop 2 - Parallel multi-perspective query generation
        self.retrieve_k10 = dspy.Retrieve(k=self.k_per_query)
        self.entity_query_hop2 = dspy.ChainOfThought(EntityBasedQuery)
        self.contrastive_query_hop2 = dspy.ChainOfThought(ContrastiveQuery)
        self.relational_query_hop2 = dspy.ChainOfThought(RelationalQuery)
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Hop 3 - Parallel multi-perspective query generation with updated context
        self.entity_query_hop3 = dspy.ChainOfThought(EntityBasedQuery)
        self.contrastive_query_hop3 = dspy.ChainOfThought(ContrastiveQuery)
        self.relational_query_hop3 = dspy.ChainOfThought(RelationalQuery)

    def _compute_novelty_scores(self, candidate_docs, existing_docs):
        """Compute novelty scores for candidate documents relative to existing documents.

        Args:
            candidate_docs: List of document strings to score
            existing_docs: List of already selected document strings

        Returns:
            Dictionary mapping doc index to novelty score (higher = more novel)
        """
        scores = {}

        # Extract document titles for comparison
        existing_titles = set()
        for doc in existing_docs:
            title = doc.split(" | ")[0].lower().strip()
            existing_titles.add(title)

        # Score each candidate based on title uniqueness and content diversity
        for idx, doc in enumerate(candidate_docs):
            title = doc.split(" | ")[0].lower().strip()

            # Title-based novelty (primary signal)
            if title in existing_titles:
                # Duplicate title gets very low score
                title_score = 0.0
            else:
                # Novel title gets high base score
                title_score = 1.0

            # Content-based diversity score (secondary signal)
            # Use simple word overlap as a proxy for content similarity
            if len(existing_docs) > 0:
                doc_words = set(doc.lower().split())
                max_overlap = 0.0
                for existing_doc in existing_docs:
                    existing_words = set(existing_doc.lower().split())
                    if len(doc_words) > 0 and len(existing_words) > 0:
                        overlap = len(doc_words & existing_words) / len(doc_words | existing_words)
                        max_overlap = max(max_overlap, overlap)

                content_score = 1.0 - max_overlap
            else:
                content_score = 1.0

            # Combined score: title novelty is weighted heavily (0.7), content diversity adds refinement (0.3)
            scores[idx] = 0.7 * title_score + 0.3 * content_score

        return scores

    def _select_top_novel_docs(self, candidate_docs, existing_docs, k):
        """Select top k most novel documents from candidates.

        Args:
            candidate_docs: List of candidate document strings
            existing_docs: List of already selected document strings
            k: Number of documents to select

        Returns:
            List of k most novel documents
        """
        if len(candidate_docs) == 0:
            return []

        # Deduplicate candidates by title first
        seen_titles = set()
        unique_candidates = []
        for doc in candidate_docs:
            title = doc.split(" | ")[0].lower().strip()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_candidates.append(doc)

        # Compute novelty scores
        scores = self._compute_novelty_scores(unique_candidates, existing_docs)

        # Sort by score (descending) and take top k
        sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        top_k_indices = sorted_indices[:k]

        return [unique_candidates[idx] for idx in top_k_indices]

    def forward(self, claim):
        # HOP 1: Direct retrieval from claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2: Parallel multi-perspective retrieval
        context_hop2 = summary_1

        # Generate 3 diverse queries in parallel
        entity_query = self.entity_query_hop2(claim=claim, context=context_hop2).query
        contrastive_query = self.contrastive_query_hop2(claim=claim, context=context_hop2).query
        relational_query = self.relational_query_hop2(claim=claim, context=context_hop2).query

        # Retrieve k=10 docs for each query (30 total)
        entity_docs = self.retrieve_k10(entity_query).passages
        contrastive_docs = self.retrieve_k10(contrastive_query).passages
        relational_docs = self.retrieve_k10(relational_query).passages

        # Combine all retrieved docs and select top 7 most novel
        hop2_candidates = entity_docs + contrastive_docs + relational_docs
        hop2_docs = self._select_top_novel_docs(hop2_candidates, hop1_docs, self.k_final)

        # Summarize hop2 results
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Parallel multi-perspective retrieval with updated context
        context_hop3 = summary_1 + " " + summary_2

        # Generate 3 diverse queries with updated context
        entity_query_3 = self.entity_query_hop3(claim=claim, context=context_hop3).query
        contrastive_query_3 = self.contrastive_query_hop3(claim=claim, context=context_hop3).query
        relational_query_3 = self.relational_query_hop3(claim=claim, context=context_hop3).query

        # Retrieve k=10 docs for each query (30 total)
        entity_docs_3 = self.retrieve_k10(entity_query_3).passages
        contrastive_docs_3 = self.retrieve_k10(contrastive_query_3).passages
        relational_docs_3 = self.retrieve_k10(relational_query_3).passages

        # Combine all retrieved docs and select top 7 most novel from hops 1+2
        hop3_candidates = entity_docs_3 + contrastive_docs_3 + relational_docs_3
        existing_docs = hop1_docs + hop2_docs
        hop3_docs = self._select_top_novel_docs(hop3_candidates, existing_docs, self.k_final)

        # Return 21 documents total (7 from each hop)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
