import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# New Signature Classes for Diversity-Aware Iterative Retrieval

class ClaimDecomposer(dspy.Signature):
    """Decompose a claim into 2-3 diverse sub-queries targeting different entities or aspects of the claim."""

    claim: str = dspy.InputField()
    sub_queries: list[str] = dspy.OutputField(
        desc="2-3 sub-queries, each targeting different entities or aspects of the claim"
    )


class CoverageAnalyzer(dspy.Signature):
    """Identify which entities or concepts are missing from the current retrieved documents."""

    claim: str = dspy.InputField()
    retrieved_passages: str = dspy.InputField(desc="current retrieved document passages")
    missing_entities: str = dspy.OutputField(
        desc="entities, concepts, or aspects missing from current documents but needed to verify the claim"
    )


class GapQuery(dspy.Signature):
    """Generate a targeted query to retrieve missing information identified in coverage analysis."""

    claim: str = dspy.InputField()
    missing_entities: str = dspy.InputField(desc="entities or concepts missing from current documents")
    gap_query: str = dspy.OutputField(
        desc="a focused query to retrieve documents covering the missing information"
    )


class DiversityReranker(dspy.Signature):
    """Select the top 21 most diverse and relevant documents using MMR-style scoring."""

    claim: str = dspy.InputField()
    all_passages: str = dspy.InputField(desc="all retrieved document passages")
    selected_indices: list[int] = dspy.OutputField(
        desc="indices of the 21 most diverse and relevant documents (0-indexed)"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using Diversity-Aware Iterative Retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize new diversity-aware components
        self.decomposer = dspy.ChainOfThought(ClaimDecomposer)
        self.coverage_analyzer = dspy.ChainOfThought(CoverageAnalyzer)
        self.gap_query_generator = dspy.ChainOfThought(GapQuery)
        self.diversity_reranker = dspy.ChainOfThought(DiversityReranker)

        # Retrieval modules with different k values
        self.retrieve_k10 = dspy.Retrieve(k=10)
        self.retrieve_k15 = dspy.Retrieve(k=15)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Decompose claim into 2-3 diverse sub-queries
            decomposition = self.decomposer(claim=claim)
            sub_queries = decomposition.sub_queries

            # Ensure we have at least 1 sub-query, cap at 3
            if not sub_queries or len(sub_queries) == 0:
                sub_queries = [claim]
            elif len(sub_queries) > 3:
                sub_queries = sub_queries[:3]

            # Step 2: Hop 1 - Retrieve k=10 documents per sub-query in parallel
            hop1_docs_all = []
            for sub_query in sub_queries:
                docs = self.retrieve_k10(sub_query).passages
                hop1_docs_all.extend(docs)

            # Remove duplicates while preserving order
            seen = set()
            hop1_docs_unique = []
            for doc in hop1_docs_all:
                # Use document content as key for deduplication
                if doc not in seen:
                    seen.add(doc)
                    hop1_docs_unique.append(doc)

            # Step 3: Analyze coverage gaps in the retrieved documents
            retrieved_passages_str = "\n---\n".join(hop1_docs_unique)
            coverage = self.coverage_analyzer(
                claim=claim,
                retrieved_passages=retrieved_passages_str
            )
            missing_entities = coverage.missing_entities

            # Step 4: Hop 2 - Generate gap-filling query and retrieve k=15 documents
            gap_query_result = self.gap_query_generator(
                claim=claim,
                missing_entities=missing_entities
            )
            gap_query = gap_query_result.gap_query

            hop2_docs = self.retrieve_k15(gap_query).passages

            # Step 5: Combine all documents and deduplicate
            all_docs = hop1_docs_unique + hop2_docs
            seen_combined = set()
            all_docs_unique = []
            for doc in all_docs:
                if doc not in seen_combined:
                    seen_combined.add(doc)
                    all_docs_unique.append(doc)

            # Step 6: Apply diversity reranking to select top 21 documents
            # If we have <= 21 documents, return them all
            if len(all_docs_unique) <= 21:
                final_docs = all_docs_unique
            else:
                # Use MMR-style diversity reranking
                all_passages_str = "\n---\n".join([
                    f"[{i}] {doc}" for i, doc in enumerate(all_docs_unique)
                ])

                reranking = self.diversity_reranker(
                    claim=claim,
                    all_passages=all_passages_str
                )
                selected_indices = reranking.selected_indices

                # Ensure we have valid indices
                valid_indices = [
                    idx for idx in selected_indices
                    if isinstance(idx, int) and 0 <= idx < len(all_docs_unique)
                ]

                # If we got valid indices, use them; otherwise take first 21
                if valid_indices and len(valid_indices) > 0:
                    # Take up to 21 indices
                    final_indices = valid_indices[:21]
                    final_docs = [all_docs_unique[idx] for idx in final_indices]
                else:
                    # Fallback: take first 21 documents
                    final_docs = all_docs_unique[:21]

            # Ensure exactly 21 documents (pad if needed, truncate if over)
            if len(final_docs) < 21:
                final_docs = final_docs  # Return what we have
            elif len(final_docs) > 21:
                final_docs = final_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
