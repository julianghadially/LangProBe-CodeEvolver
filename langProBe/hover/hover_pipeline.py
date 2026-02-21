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

        # Listwise reranker module
        self.reranker = dspy.ChainOfThought(ListwiseReranker)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []

            # Stage 1: Initial retrieval with k=100 documents using the original claim query
            initial_docs = self.initial_retrieve(claim).passages
            all_docs.extend(initial_docs)

            # Stage 2: Extract key entities from the top 50 initial results
            top_50_docs = initial_docs[:50]
            docs_text = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(top_50_docs)])

            try:
                entity_result = self.entity_extractor(claim=claim, documents=docs_text)
                entities = entity_result.entities

                # Limit to top 3 entities to stay within search limit
                entities = entities[:3] if isinstance(entities, list) else []

                # Stage 3: Generate focused queries for each extracted entity and retrieve k=50 documents per entity
                for entity in entities:
                    try:
                        entity_query_result = self.entity_query_gen(claim=claim, entity=entity)
                        entity_query = entity_query_result.query
                        entity_docs = self.entity_retrieve(entity_query).passages
                        all_docs.extend(entity_docs)
                    except Exception:
                        # If entity query generation or retrieval fails, continue with next entity
                        continue

            except Exception:
                # If entity extraction fails, continue with just the initial docs
                pass

            # Stage 4: Combine all retrieved documents (initial + entity-based)
            # Deduplicate by document title while preserving order
            seen_titles = set()
            unique_docs = []
            for doc in all_docs:
                title = doc.split(" | ")[0]
                normalized_title = title.lower().strip()
                if normalized_title not in seen_titles:
                    seen_titles.add(normalized_title)
                    unique_docs.append(doc)

            # Stage 5: Apply listwise reranker using ChainOfThought reasoning
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
