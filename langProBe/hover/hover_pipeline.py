import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# ============ New DSPy Signature Classes for Query Decomposition Architecture ============

class ClaimDecomposition(dspy.Signature):
    """Decompose a complex multi-hop claim into 2-3 specific sub-questions that can be answered independently.
    Each sub-question should target a distinct piece of information needed to verify the claim."""

    claim: str = dspy.InputField(desc="the complex claim to verify")
    sub_questions: list[str] = dspy.OutputField(desc="2-3 specific sub-questions that break down the claim into answerable components")


class EntityExtractor(dspy.Signature):
    """Extract key entities and relationships from retrieved documents that are relevant to verifying the claim.
    Focus on concrete entities (people, places, organizations, dates, events) and their relationships."""

    claim: str = dspy.InputField(desc="the claim being verified")
    documents: str = dspy.InputField(desc="the retrieved documents to analyze")
    entities: list[str] = dspy.OutputField(desc="list of key entities discovered (e.g., 'Person: John Doe', 'Movie: The River Rat', 'Date: 1984')")
    relationships: list[str] = dspy.OutputField(desc="list of relationships between entities (e.g., 'John Doe directed The River Rat', 'The River Rat was released in 1984')")


class GapAnalysis(dspy.Signature):
    """Analyze what critical information is still missing to verify the claim, given what has been discovered so far.
    Identify specific entities, relationships, or facts that need to be retrieved in the next iteration."""

    claim: str = dspy.InputField(desc="the claim being verified")
    entities_found: str = dspy.InputField(desc="entities discovered so far")
    relationships_found: str = dspy.InputField(desc="relationships discovered so far")
    documents_retrieved: str = dspy.InputField(desc="summary of documents retrieved so far")
    missing_information: list[str] = dspy.OutputField(desc="specific pieces of missing information needed to verify the claim")
    targeted_queries: list[str] = dspy.OutputField(desc="2-3 specific search queries to find the missing information")


class QueryPerspectiveGenerator(dspy.Signature):
    """Generate 4-5 diverse query reformulations from different perspectives to maximize retrieval coverage.
    Each query should approach the claim from a unique angle to find supporting documents that might be missed by a single query.

    Perspectives to consider:
    - Entity-focused: queries targeting specific people, places, or organizations mentioned
    - Relationship-focused: queries about connections and interactions between entities
    - Temporal-focused: queries about time periods, dates, or chronological aspects
    - Comparison-focused: queries comparing or contrasting different aspects of the claim
    - Context-focused: queries about broader context or background information
    """

    claim: str = dspy.InputField(desc="the claim to verify")
    context: str = dspy.InputField(desc="additional context from previously retrieved documents")
    query_perspectives: list[str] = dspy.OutputField(desc="4-5 diverse query reformulations from different perspectives (entity-focused, relationship-focused, temporal-focused, comparison-focused, context-focused)")


# ============ Main Pipeline with Iterative Entity Discovery ============

class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim using Reciprocal Rank Fusion (RRF) reranking.

    ARCHITECTURE:
    - Uses query perspective generation to create 4-5 diverse query reformulations per iteration
    - Retrieves k=7 documents per query (~35 docs per iteration)
    - Applies RRF formula: score(doc) = sum(1/(60 + rank_in_query_i)) across all queries
    - Keeps top 30 documents after each iteration based on RRF scores
    - After 3 iterations, applies final RRF scoring and returns top 21 documents
    - No LLM-based scoring; RRF mathematically prioritizes documents with consistent ranking

    EVALUATION:
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize modules for RRF architecture
        self.perspective_generator = dspy.ChainOfThought(QueryPerspectiveGenerator)
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysis)

        # Retrieval module: k=7 documents per query
        self.retrieve = dspy.Retrieve(k=7)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Helper function to normalize DSPy outputs to lists
            def normalize_list(value):
                """Helper to normalize DSPy outputs to lists."""
                if not isinstance(value, list):
                    if isinstance(value, str):
                        items = [q.strip() for q in value.split('\n') if q.strip()]
                        items = [q.lstrip('0123456789.-)> ').strip() for q in items if q.strip()]
                        return items
                    return [str(value)] if value else []
                return value

            # Helper function to apply RRF scoring
            def apply_rrf(query_results):
                """Apply Reciprocal Rank Fusion across multiple query results.

                Args:
                    query_results: List of lists, where each inner list is ranked documents from one query

                Returns:
                    List of (doc, rrf_score) tuples sorted by RRF score descending
                """
                rrf_scores = {}  # doc -> cumulative RRF score

                for query_docs in query_results:
                    for rank, doc in enumerate(query_docs):
                        # RRF formula: score(doc) = sum(1/(60 + rank))
                        # rank is 0-indexed, so rank 0 gets 1/60, rank 1 gets 1/61, etc.
                        rrf_score = 1.0 / (60 + rank)
                        if doc in rrf_scores:
                            rrf_scores[doc] += rrf_score
                        else:
                            rrf_scores[doc] = rrf_score

                # Sort by RRF score descending
                sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_docs

            # Storage for all query results across all iterations
            all_query_results = []  # List of lists: each element is a ranked list of docs from one query
            context_summary = ""  # Track context for next iteration

            # ========== ITERATION 1: Initial Query Perspectives ==========

            # Generate 4-5 diverse query perspectives
            try:
                perspective_result = self.perspective_generator(claim=claim, context=claim)
                queries_iter1 = normalize_list(perspective_result.query_perspectives)[:5]  # Max 5
            except Exception:
                queries_iter1 = [claim]

            # Retrieve k=7 documents per query
            for query in queries_iter1:
                try:
                    docs = self.retrieve(query).passages
                    all_query_results.append(docs)
                except Exception:
                    pass

            # Apply RRF after iteration 1 to get top 30 documents
            if all_query_results:
                rrf_ranked = apply_rrf(all_query_results)
                top30_iter1 = [doc for doc, score in rrf_ranked[:30]]
                context_summary = "\n\n".join(top30_iter1[:10])  # Use top 10 for context
            else:
                top30_iter1 = []
                context_summary = claim

            # ========== ITERATION 2: Query Perspectives with Context ==========

            # Generate 4-5 diverse query perspectives informed by iteration 1 results
            try:
                perspective_result2 = self.perspective_generator(claim=claim, context=context_summary)
                queries_iter2 = normalize_list(perspective_result2.query_perspectives)[:5]  # Max 5
            except Exception:
                queries_iter2 = [claim]

            # Retrieve k=7 documents per query
            for query in queries_iter2:
                try:
                    docs = self.retrieve(query).passages
                    all_query_results.append(docs)
                except Exception:
                    pass

            # Apply RRF after iteration 2 to get top 30 documents
            if all_query_results:
                rrf_ranked = apply_rrf(all_query_results)
                top30_iter2 = [doc for doc, score in rrf_ranked[:30]]
                context_summary = "\n\n".join(top30_iter2[:10])  # Use top 10 for context
            else:
                top30_iter2 = []
                context_summary = claim

            # ========== ITERATION 3: Final Query Perspectives ==========

            # Generate 4-5 diverse query perspectives informed by iteration 2 results
            try:
                perspective_result3 = self.perspective_generator(claim=claim, context=context_summary)
                queries_iter3 = normalize_list(perspective_result3.query_perspectives)[:5]  # Max 5
            except Exception:
                queries_iter3 = [claim]

            # Retrieve k=7 documents per query
            for query in queries_iter3:
                try:
                    docs = self.retrieve(query).passages
                    all_query_results.append(docs)
                except Exception:
                    pass

            # ========== FINAL RRF SCORING: Apply RRF across all accumulated query results ==========

            if all_query_results:
                final_rrf_ranked = apply_rrf(all_query_results)
                top21_docs = [doc for doc, score in final_rrf_ranked[:21]]
            else:
                # Fallback: if no results, return empty list
                top21_docs = []

            return dspy.Prediction(retrieved_docs=top21_docs)
