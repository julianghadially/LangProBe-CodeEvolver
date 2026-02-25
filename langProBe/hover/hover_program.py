import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractEntities(dspy.Signature):
    """Extract 3-5 specific named entities (people, works, places, organizations) from a claim.
    Focus on concrete entities that can be used to retrieve factual information, not abstract concepts."""

    claim: str = dspy.InputField(desc="The claim to extract entities from")
    entities: list[str] = dspy.OutputField(desc="3-5 specific named entities (people, works, places, organizations) from the claim")


class EntityToQuery(dspy.Signature):
    """Generate a targeted search query focused on retrieving documents about a specific entity in the context of a claim.
    The query should be entity-specific and designed to find factual information relevant to verifying the claim."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    entity: str = dspy.InputField(desc="The specific entity to focus the query on")
    query: str = dspy.OutputField(desc="A targeted search query to retrieve documents about this entity")


class LLMReranker(dspy.Signature):
    """Analyze a batch of candidate documents and score their relevance to a claim.
    Output relevance scores from 1-10 for each document, where 10 is highly relevant and 1 is not relevant.
    Consider whether the document contains specific facts needed to verify the claim."""

    claim: str = dspy.InputField(desc="The claim being fact-checked")
    documents: str = dspy.InputField(desc="Batch of candidate documents to score, formatted as numbered list")
    scores: list[int] = dspy.OutputField(desc="List of relevance scores (1-10) for each document in order")


class ClaimDecomposition(dspy.Signature):
    """Decompose a claim into 2-3 focused sub-queries that target different entities or concepts within the claim.
    Each sub-query should focus on a distinct entity, concept, or aspect of the claim to enable parallel retrieval
    of documents about all entities mentioned rather than following a single sequential path."""

    claim: str = dspy.InputField(desc="The claim to decompose into sub-queries")
    sub_queries: list[str] = dspy.OutputField(desc="2-3 focused sub-queries, each targeting a different entity or concept in the claim")


class RelevanceScorer(dspy.Signature):
    """Score a document's relevance to the original claim on a 1-10 scale with reasoning.
    Provide chain-of-thought reasoning explaining why the document is or isn't relevant to verifying the claim."""

    claim: str = dspy.InputField(desc="The original claim being fact-checked")
    document: str = dspy.InputField(desc="The document to score for relevance")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning about the document's relevance to the claim")
    score: int = dspy.OutputField(desc="Relevance score from 1-10, where 10 is highly relevant and 1 is not relevant")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using entity-focused queries and LLM-based reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Entity-focused retrieval modules
        self.extract_entities = dspy.Predict(ExtractEntities)
        self.entity_to_query = dspy.Predict(EntityToQuery)
        self.retrieve_k = dspy.Retrieve(k=50)  # Retrieve 50 documents per entity query

        # LLM-based reranking module
        self.reranker = dspy.ChainOfThought(LLMReranker)

    def forward(self, claim):
        # Step 1: Extract 3-5 specific named entities from the claim
        entity_result = self.extract_entities(claim=claim)
        entities = entity_result.entities

        # Ensure we have a list of entities
        if not isinstance(entities, list):
            entities = [entities]

        # Limit to 3 entities to retrieve max 150 documents (3 * 50)
        entities = entities[:3]

        # Handle edge case: if no entities extracted, use the claim itself
        if len(entities) == 0:
            entities = [claim]

        # Step 2: For each entity, generate a targeted search query
        entity_queries = []
        for entity in entities:
            try:
                query_result = self.entity_to_query(claim=claim, entity=entity)
                entity_queries.append(query_result.query)
            except (AttributeError, ValueError):
                # Fallback: use entity directly as query
                entity_queries.append(entity)

        # Step 3: Retrieve k=50 documents per entity query (up to 150 total)
        all_docs = []
        for query in entity_queries:
            try:
                docs = self.retrieve_k(query).passages
                all_docs.extend(docs)
            except Exception:
                # Skip if retrieval fails for this query
                continue

        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_key = doc.lower().strip()
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)

        # Step 4: Apply LLM reranking in sliding windows of 10 documents
        batch_size = 10
        scored_docs = []

        for i in range(0, len(unique_docs), batch_size):
            batch = unique_docs[i:i+batch_size]

            # Format documents as numbered list for the reranker
            doc_list = "\n".join([f"{j+1}. {doc}" for j, doc in enumerate(batch)])

            try:
                rerank_result = self.reranker(claim=claim, documents=doc_list)
                scores = rerank_result.scores

                # Ensure scores is a list
                if not isinstance(scores, list):
                    scores = [scores]

                # Match scores to documents
                for j, doc in enumerate(batch):
                    if j < len(scores):
                        try:
                            score = int(scores[j]) if isinstance(scores[j], (int, str)) else 5
                            score = max(1, min(10, score))  # Clamp to 1-10 range
                        except (ValueError, TypeError):
                            score = 5  # Default score if parsing fails
                    else:
                        score = 5  # Default score if not enough scores returned

                    scored_docs.append((doc, score))

            except (AttributeError, ValueError, Exception):
                # If reranking fails, assign default scores to this batch
                for doc in batch:
                    scored_docs.append((doc, 5))

        # Step 5: Select top 21 documents based on LLM reranking scores
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by title and select top 21
        seen_titles = set()
        final_docs = []

        for doc, score in scored_docs:
            # Extract title (before the " | " separator)
            title = doc.split(" | ")[0] if " | " in doc else doc
            normalized_title = title.lower().strip()

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                final_docs.append(doc)

                # Stop once we have 21 unique documents
                if len(final_docs) >= 21:
                    break

        return dspy.Prediction(retrieved_docs=final_docs)
