import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# New Signature Classes for Recursive Query Expansion

class InitialQueryGenerator(dspy.Signature):
    """Generate an initial search query to find documents relevant to verifying the claim."""

    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="an effective search query to retrieve documents relevant to the claim")


class EntityExtractor(dspy.Signature):
    """Extract entity mentions from retrieved document titles that are relevant to the claim but not yet fully explored."""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="titles of documents retrieved so far")
    unexplored_entities: list[str] = dspy.OutputField(
        desc="list of 2-5 entity mentions from titles that are relevant to the claim but need further exploration"
    )


class EntityQueryGenerator(dspy.Signature):
    """Generate a focused query to retrieve more information about a specific entity related to the claim."""

    claim: str = dspy.InputField()
    entity: str = dspy.InputField(desc="the entity to explore further")
    focused_query: str = dspy.OutputField(
        desc="a targeted query to retrieve documents about this entity in the context of the claim"
    )


class RelevanceReranker(dspy.Signature):
    """Score all documents by relevance to the claim and return the top 21 most relevant documents."""

    claim: str = dspy.InputField()
    documents: str = dspy.InputField(desc="all retrieved documents with indices")
    top_indices: list[int] = dspy.OutputField(
        desc="indices of the top 21 most relevant documents, ordered by relevance (0-indexed)"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using Recursive Query Expansion.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize Recursive Query Expansion components
        self.initial_query_generator = dspy.ChainOfThought(InitialQueryGenerator)
        self.entity_extractor = dspy.ChainOfThought(EntityExtractor)
        self.entity_query_generator = dspy.ChainOfThought(EntityQueryGenerator)
        self.relevance_reranker = dspy.ChainOfThought(RelevanceReranker)

        # Retrieval modules with different k values
        self.retrieve_k10 = dspy.Retrieve(k=10)
        self.retrieve_k15 = dspy.Retrieve(k=15)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Hop 1 - Generate initial query and retrieve k=15 docs
            initial_query_result = self.initial_query_generator(claim=claim)
            initial_query = initial_query_result.query
            hop1_docs = self.retrieve_k15(initial_query).passages

            # Track all documents and their titles
            all_docs = list(hop1_docs)

            # Extract document titles from hop 1 (format: "Title | text...")
            hop1_titles = [doc.split(" | ")[0] if " | " in doc else doc[:100] for doc in hop1_docs]
            retrieved_titles_str = "\n".join(hop1_titles)

            # Step 2: Hop 2 - Extract entity mentions from retrieved titles
            entity_extraction = self.entity_extractor(
                claim=claim,
                retrieved_titles=retrieved_titles_str
            )
            unexplored_entities = entity_extraction.unexplored_entities

            # Ensure we have a list of entities
            if not isinstance(unexplored_entities, list):
                unexplored_entities = []

            # Generate targeted queries for top 2-3 unexplored entities
            entities_to_explore = unexplored_entities[:3] if len(unexplored_entities) >= 3 else unexplored_entities[:2]

            for entity in entities_to_explore:
                # Generate focused query for this entity
                entity_query_result = self.entity_query_generator(
                    claim=claim,
                    entity=entity
                )
                entity_query = entity_query_result.focused_query

                # Retrieve k=10 docs for this entity
                entity_docs = self.retrieve_k10(entity_query).passages
                all_docs.extend(entity_docs)

            # Step 3: Check if under 50 total docs, if so do hop 3
            # Deduplicate first to get accurate count
            seen = set()
            all_docs_unique = []
            for doc in all_docs:
                if doc not in seen:
                    seen.add(doc)
                    all_docs_unique.append(doc)

            if len(all_docs_unique) < 50:
                # Hop 3 - Extract entities again and do one more retrieval
                all_titles = [doc.split(" | ")[0] if " | " in doc else doc[:100] for doc in all_docs_unique]
                all_titles_str = "\n".join(all_titles)

                hop3_entity_extraction = self.entity_extractor(
                    claim=claim,
                    retrieved_titles=all_titles_str
                )
                hop3_entities = hop3_entity_extraction.unexplored_entities

                if not isinstance(hop3_entities, list):
                    hop3_entities = []

                # Take first entity for hop 3 retrieval
                if hop3_entities:
                    hop3_entity = hop3_entities[0]
                    hop3_query_result = self.entity_query_generator(
                        claim=claim,
                        entity=hop3_entity
                    )
                    hop3_query = hop3_query_result.focused_query
                    hop3_docs = self.retrieve_k15(hop3_query).passages
                    all_docs.extend(hop3_docs)

                    # Deduplicate again after hop 3
                    for doc in hop3_docs:
                        if doc not in seen:
                            seen.add(doc)
                            all_docs_unique.append(doc)

            # Step 4: Use RelevanceReranker to score and select top 21 docs
            if len(all_docs_unique) <= 21:
                final_docs = all_docs_unique
            else:
                # Prepare documents with indices for reranking
                documents_str = "\n---\n".join([
                    f"[{i}] {doc}" for i, doc in enumerate(all_docs_unique)
                ])

                reranking = self.relevance_reranker(
                    claim=claim,
                    documents=documents_str
                )
                top_indices = reranking.top_indices

                # Ensure we have valid indices
                valid_indices = [
                    idx for idx in top_indices
                    if isinstance(idx, int) and 0 <= idx < len(all_docs_unique)
                ]

                # Select top 21 by relevance score
                if valid_indices and len(valid_indices) > 0:
                    final_indices = valid_indices[:21]
                    final_docs = [all_docs_unique[idx] for idx in final_indices]
                else:
                    # Fallback: take first 21 documents
                    final_docs = all_docs_unique[:21]

            # Ensure we return up to 21 documents
            if len(final_docs) > 21:
                final_docs = final_docs[:21]

            return dspy.Prediction(retrieved_docs=final_docs)
