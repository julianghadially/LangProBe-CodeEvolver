import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractNamedEntities(dspy.Signature):
    """Extract named entities and key descriptive phrases from the claim that need direct retrieval. Focus on proper nouns (people, titles, places, organizations) and bridging concepts."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    named_entities: list[str] = dspy.OutputField(desc="List of proper nouns and named entities (people, book/film/work titles, places, organizations) found in the claim")
    descriptive_phrases: list[str] = dspy.OutputField(desc="List of key descriptive phrases or concepts that need bridging documents for multi-hop reasoning")


class EntityExtraction(dspy.Signature):
    """Extract key entities (people, places, works, organizations) from retrieved documents that are relevant to verifying the claim."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    documents: str = dspy.InputField(desc="Retrieved documents to extract entities from")
    entities: list[str] = dspy.OutputField(desc="List of 1-5 key entities (people, places, works, organizations) most relevant to the claim")


class ContextualQueryGenerator(dspy.Signature):
    """Generate 1-2 contextual queries that combine entities with key relationships from the claim to find bridging documents for multi-hop reasoning."""
    claim: str = dspy.InputField(desc="The claim to be verified")
    entities: str = dspy.InputField(desc="The named entities extracted from the claim")
    queries: list[str] = dspy.OutputField(desc="1-2 contextual search queries combining entities with relationships from the claim")


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


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize sub-modules for the new two-stage retrieval strategy
        # Stage 1: Entity retrieval with k=30
        self.entity_retrieve = dspy.Retrieve(k=30)
        # Stage 2: Contextual retrieval with k=20
        self.contextual_retrieve = dspy.Retrieve(k=20)

        # Named entity extraction from claim (not documents)
        self.named_entity_extractor = dspy.ChainOfThought(ExtractNamedEntities)
        # Contextual query generation combining entities with relationships
        self.contextual_query_gen = dspy.ChainOfThought(ContextualQueryGenerator)

        # Listwise reranker module
        self.reranker = dspy.ChainOfThought(ListwiseReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []

            # Extract named entities and descriptive phrases directly from the claim
            try:
                entity_extraction_result = self.named_entity_extractor(claim=claim)
                named_entities = entity_extraction_result.named_entities if isinstance(entity_extraction_result.named_entities, list) else []
                descriptive_phrases = entity_extraction_result.descriptive_phrases if isinstance(entity_extraction_result.descriptive_phrases, list) else []
            except Exception:
                # If extraction fails, fall back to empty lists
                named_entities = []
                descriptive_phrases = []

            # Stage 1 - Entity Retrieval: Direct retrieval with k=30 for each named entity
            # Batch all entity names to stay within search limit (combine entities + descriptive phrases)
            all_entity_terms = named_entities + descriptive_phrases

            # Retrieve documents for each entity/phrase directly using their names
            for entity_term in all_entity_terms:
                try:
                    # Use entity name directly as query (no query generation)
                    entity_docs = self.entity_retrieve(entity_term).passages
                    all_docs.extend(entity_docs)
                except Exception:
                    # If retrieval fails for this entity, continue with next
                    continue

            # Stage 2 - Contextual Retrieval: Generate 1-2 contextual queries combining entities with relationships
            try:
                # Prepare entities string for contextual query generation
                entities_str = ", ".join(all_entity_terms[:5])  # Use top 5 entities to avoid overly long input

                contextual_query_result = self.contextual_query_gen(claim=claim, entities=entities_str)
                contextual_queries = contextual_query_result.queries if isinstance(contextual_query_result.queries, list) else []

                # Limit to 2 contextual queries as specified
                contextual_queries = contextual_queries[:2]

                # Retrieve k=20 documents per contextual query
                for query in contextual_queries:
                    try:
                        contextual_docs = self.contextual_retrieve(query).passages
                        all_docs.extend(contextual_docs)
                    except Exception:
                        # If retrieval fails for this query, continue with next
                        continue

            except Exception:
                # If contextual query generation fails, continue without contextual docs
                pass

            # Merge and deduplicate all retrieved documents
            # Deduplicate by document title while preserving order
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0]
                normalized_title = title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Apply listwise LLM reranker
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
