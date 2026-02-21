import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtraction(dspy.Signature):
    """Extract key entities (people, places, works, organizations) from retrieved documents that are relevant to verifying the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Retrieved documents to extract entities from")
    entities: list[str] = dspy.OutputField(desc="List of 1-5 key entities (people, places, works, organizations) most relevant to the claim")


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused search query for a specific entity in the context of the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    entity: str = dspy.InputField(desc="The entity to generate a query for")
    query: str = dspy.OutputField(desc="A focused search query for the entity")


class ListwiseReranker(dspy.Signature):
    """Score and rank documents based on their relevance to the multi-hop reasoning chain needed to verify the claim. Consider how documents connect to support multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Documents to rank, each prefixed with an index like [0], [1], etc.")
    top_indices: list[int] = dspy.OutputField(desc="List of document indices ranked by relevance (most relevant first), selecting the top 21 documents")
    reasoning: str = dspy.OutputField(desc="Explanation of the multi-hop reasoning chain and why these documents are most relevant")


class CoverageVerifier(dspy.Signature):
    """Verify if the currently retrieved documents adequately cover all aspects and entities needed to verify the claim. Assess coverage quality and identify gaps."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    current_doc_titles: str = dspy.InputField(desc="Titles of currently retrieved documents, one per line")
    coverage_score: float = dspy.OutputField(desc="Coverage score from 0.0 to 1.0 indicating how well current documents cover claim entities and aspects")
    missing_aspects: list[str] = dspy.OutputField(desc="List of claim aspects, entities, or information gaps not yet covered by current documents")
    confidence: bool = dspy.OutputField(desc="Whether coverage is sufficient (True) or additional retrieval is needed (False)")


class AdaptiveQueryGenerator(dspy.Signature):
    """Generate alternative search queries that target missing aspects using different phrasings, angles, or perspectives to improve retrieval coverage."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    missing_aspects: str = dspy.InputField(desc="Aspects or entities that need coverage, formatted as bullet points")
    already_tried_queries: str = dspy.InputField(desc="Queries already attempted, one per line")
    alternative_queries: list[str] = dspy.OutputField(desc="1-2 alternative search queries targeting missing aspects with different phrasings or angles")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize sub-modules for the two-stage retrieval strategy
        self.initial_retrieve = dspy.Retrieve(k=100)
        self.entity_retrieve = dspy.Retrieve(k=50)

        # Entity extraction and query generation modules
        self.entity_extractor = dspy.ChainOfThought(EntityExtraction)
        self.entity_query_gen = dspy.ChainOfThought(EntityQueryGenerator)

        # Self-verification loop modules
        self.coverage_verifier = dspy.ChainOfThought(CoverageVerifier)
        self.adaptive_query_gen = dspy.ChainOfThought(AdaptiveQueryGenerator)

        # Listwise reranker module
        self.reranker = dspy.ChainOfThought(ListwiseReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []
            tried_queries = [claim]  # Track all queries attempted
            max_iterations = 2  # Maximum additional retrieval iterations (total 3 searches)

            # Stage 1: Initial retrieval with k=100 documents using the original claim query
            initial_docs = self.initial_retrieve(claim).passages
            all_docs.extend(initial_docs)

            # Deduplicate documents by title and get current coverage
            def deduplicate_docs(docs):
                """Deduplicate documents by normalized title while preserving order."""
                seen_titles = set()
                unique = []
                for doc in docs:
                    title = doc.split(" | ")[0]
                    normalized_title = title.lower().strip()
                    if normalized_title not in seen_titles:
                        seen_titles.add(normalized_title)
                        unique.append(doc)
                return unique

            # Stage 2: Self-Verification Loop with Adaptive Query Refinement
            for iteration in range(max_iterations):
                # Deduplicate current documents
                unique_docs = deduplicate_docs(all_docs)

                # Extract document titles for coverage verification
                doc_titles = "\n".join([doc.split(" | ")[0] for doc in unique_docs[:100]])  # Top 100 for analysis

                try:
                    # Verify coverage of current retrieval
                    coverage_result = self.coverage_verifier(
                        claim=claim,
                        current_doc_titles=doc_titles
                    )

                    # If coverage is sufficient, break early
                    if coverage_result.confidence:
                        break

                    # If coverage is insufficient and we have iterations left, generate adaptive queries
                    missing_aspects = coverage_result.missing_aspects
                    if not missing_aspects or not isinstance(missing_aspects, list):
                        # No clear missing aspects identified, break loop
                        break

                    # Format missing aspects and tried queries for adaptive generation
                    missing_aspects_text = "\n".join([f"- {aspect}" for aspect in missing_aspects])
                    tried_queries_text = "\n".join(tried_queries)

                    # Generate alternative queries targeting missing aspects
                    adaptive_result = self.adaptive_query_gen(
                        claim=claim,
                        missing_aspects=missing_aspects_text,
                        already_tried_queries=tried_queries_text
                    )

                    alternative_queries = adaptive_result.alternative_queries
                    if not alternative_queries or not isinstance(alternative_queries, list):
                        # No alternative queries generated, break loop
                        break

                    # Limit to 2 queries to stay within retrieval budget
                    alternative_queries = alternative_queries[:2]

                    # Retrieve documents for each alternative query (k=50 each)
                    for alt_query in alternative_queries:
                        if alt_query and isinstance(alt_query, str):
                            try:
                                adaptive_docs = self.entity_retrieve(alt_query).passages
                                all_docs.extend(adaptive_docs)
                                tried_queries.append(alt_query)
                            except Exception:
                                # If adaptive retrieval fails, continue
                                continue

                except Exception:
                    # If verification or adaptive generation fails, break loop
                    break

            # Stage 3: Final deduplication and reranking
            unique_docs = deduplicate_docs(all_docs)

            # Apply listwise reranker using ChainOfThought reasoning
            # Format documents with indices for the reranker
            indexed_docs = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(unique_docs)])

            try:
                rerank_result = self.reranker(claim=claim, documents=indexed_docs)
                top_indices = rerank_result.top_indices

                # Ensure indices are valid and limit to 21
                valid_indices = []
                for idx in top_indices:
                    if isinstance(idx, int) and 0 <= idx < len(unique_docs):
                        valid_indices.append(idx)
                    if len(valid_indices) >= 21:
                        break

                # Select documents based on reranked indices
                final_docs = [unique_docs[idx] for idx in valid_indices]

            except Exception:
                # If reranking fails, fall back to using first 21 unique docs
                final_docs = unique_docs[:21]

            # Ensure we have exactly up to 21 documents
            final_docs = final_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
