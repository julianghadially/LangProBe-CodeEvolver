import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimEntityExtraction(dspy.Signature):
    """Extract key named entities (people, places, organizations) from the claim that need to be verified."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    entities: list[str] = dspy.OutputField(desc="List of 3-7 key named entities (people, places, organizations) mentioned in the claim")


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


class GapAnalysis(dspy.Signature):
    """Analyze which key entities from the claim are NOT well-covered in the retrieved documents."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    claim_entities: list[str] = dspy.InputField(desc="Key entities extracted from the claim")
    documents: str = dspy.InputField(desc="Retrieved documents to analyze for entity coverage")
    missing_entities: list[str] = dspy.OutputField(desc="List of entities from the claim that are poorly covered or missing in the documents")


class TargetedQueryGenerator(dspy.Signature):
    """Generate a targeted search query to find information about a missing entity in the context of the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    missing_entity: str = dspy.InputField(desc="The entity that needs better coverage")
    query: str = dspy.OutputField(desc="A targeted search query to find documents about this entity")


class EntityAwareReranker(dspy.Signature):
    """Score and rank documents prioritizing those that mention multiple claim entities and support multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    claim_entities: list[str] = dspy.InputField(desc="Key entities from the claim")
    documents: str = dspy.InputField(desc="Documents to rank, each prefixed with an index like [0], [1], etc.")
    top_indices: list[int] = dspy.OutputField(desc="List of document indices ranked by relevance (prioritize documents mentioning multiple claim entities), selecting the top 21 documents")
    reasoning: str = dspy.OutputField(desc="Explanation of entity coverage and multi-hop reasoning chain")


class ListwiseReranker(dspy.Signature):
    """Score and rank documents based on their relevance to the multi-hop reasoning chain needed to verify the claim. Consider how documents connect to support multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Documents to rank, each prefixed with an index like [0], [1], etc.")
    top_indices: list[int] = dspy.OutputField(desc="List of document indices ranked by relevance (most relevant first), selecting the top 21 documents")
    reasoning: str = dspy.OutputField(desc="Explanation of the multi-hop reasoning chain and why these documents are most relevant")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize sub-modules for the gap analysis retrieval strategy
        self.initial_retrieve = dspy.Retrieve(k=75)  # Initial retrieval with k=50-100
        self.gap_retrieve = dspy.Retrieve(k=40)  # Gap-filling retrieval with k=30-50

        # Gap analysis modules
        self.claim_entity_extractor = dspy.ChainOfThought(ClaimEntityExtraction)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysis)
        self.targeted_query_gen = dspy.ChainOfThought(TargetedQueryGenerator)

        # Entity-aware reranker module
        self.entity_reranker = dspy.ChainOfThought(EntityAwareReranker)

        # Legacy modules (kept for compatibility)
        self.entity_extractor = dspy.ChainOfThought(EntityExtraction)
        self.entity_query_gen = dspy.ChainOfThought(EntityQueryGenerator)
        self.reranker = dspy.ChainOfThought(ListwiseReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []
            retrieval_count = 0  # Track number of retrieval calls (max 3)

            # Stage 1: Extract key named entities from the claim using LLM
            claim_entities = []
            try:
                entity_result = self.claim_entity_extractor(claim=claim)
                claim_entities = entity_result.entities if isinstance(entity_result.entities, list) else []
                # Limit to 3-7 entities
                claim_entities = claim_entities[:7]
            except Exception:
                # If entity extraction fails, continue without entities
                claim_entities = []

            # Stage 2: Initial retrieval with k=50-100 documents
            initial_docs = self.initial_retrieve(claim).passages
            retrieval_count += 1
            all_docs.extend(initial_docs)

            # Stage 3: Gap analysis - analyze which entities are NOT well-covered
            missing_entities = []
            if claim_entities:
                try:
                    # Format documents for analysis
                    docs_text = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(initial_docs)])

                    gap_result = self.gap_analyzer(
                        claim=claim,
                        claim_entities=claim_entities,
                        documents=docs_text
                    )
                    missing_entities = gap_result.missing_entities if isinstance(gap_result.missing_entities, list) else []
                except Exception:
                    # If gap analysis fails, assume all entities need coverage
                    missing_entities = claim_entities[:3]

            # Stage 4: Generate targeted follow-up queries for missing entities
            # Stage 5: Retrieve additional documents (k=30-50) for gaps
            # Limit to 2 additional retrieval calls to maintain max 3 total
            max_gap_retrievals = min(3 - retrieval_count, 2)
            missing_entities = missing_entities[:max_gap_retrievals]

            for missing_entity in missing_entities:
                if retrieval_count >= 3:
                    break

                try:
                    # Generate targeted query for missing entity
                    query_result = self.targeted_query_gen(claim=claim, missing_entity=missing_entity)
                    targeted_query = query_result.query

                    # Retrieve additional documents for this gap
                    gap_docs = self.gap_retrieve(targeted_query).passages
                    retrieval_count += 1
                    all_docs.extend(gap_docs)
                except Exception:
                    # If query generation or retrieval fails, continue with next entity
                    continue

            # Stage 6: Merge and deduplicate all retrieved documents
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0]
                normalized_title = title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Stage 7: Rerank with entity-aware reranker (prioritizes docs with multiple claim entities)
            indexed_docs = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(unique_docs)])

            try:
                rerank_result = self.entity_reranker(
                    claim=claim,
                    claim_entities=claim_entities,
                    documents=indexed_docs
                )
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
