import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ExtractKeyEntities(dspy.Signature):
    """Extract all proper nouns, names, dates, and specific terms from the claim that would be useful for targeted search."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="List of key entities including proper nouns, names, dates, company names, battle names, work titles, and other specific terms from the claim")


class GenerateContextualQuery(dspy.Signature):
    """Generate a broad contextual query that covers the main relationship in the claim."""

    claim: str = dspy.InputField()
    query: str = dspy.OutputField(desc="A broad query covering the main claim relationship")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # Initialize hybrid retrieval components
        self.entity_extractor = dspy.Predict(ExtractKeyEntities)
        self.contextual_query_generator = dspy.Predict(GenerateContextualQuery)
        self.retrieve_k10 = dspy.Retrieve(k=10)

    def _extract_doc_title(self, doc_str):
        """Extract the title from a document string (format: 'Title | Content')."""
        if isinstance(doc_str, str) and " | " in doc_str:
            return doc_str.split(" | ")[0]
        return doc_str

    def _normalize_title(self, title):
        """Normalize document title for comparison."""
        return dspy.evaluate.normalize_text(title)

    def _score_document(self, doc_str, entities, claim_keywords, rank):
        """Score a document based on entity matches, keyword coverage, and retrieval rank."""
        doc_lower = doc_str.lower()
        score = 0

        # (a) Exact entity name matches (highest priority) - 100 points per match
        entity_matches = sum(1 for entity in entities if entity.lower() in doc_lower)
        score += entity_matches * 100

        # (b) Claim keyword coverage - 10 points per keyword
        keyword_matches = sum(1 for keyword in claim_keywords if keyword.lower() in doc_lower)
        score += keyword_matches * 10

        # (c) Retrieval rank - higher rank = lower score penalty (max 30 point penalty)
        # Rank 0 = 0 penalty, Rank 29 = 30 penalty
        rank_penalty = min(rank, 30)
        score -= rank_penalty

        return score

    def _deduplicate_and_rerank(self, all_docs, entities, claim):
        """Deduplicate by title and rerank documents based on scoring criteria."""
        # Extract claim keywords (simple word tokenization)
        claim_keywords = [word for word in claim.split() if len(word) > 3]

        # Deduplicate by title while preserving order and score
        seen_titles = {}
        unique_docs = []

        for idx, doc in enumerate(all_docs):
            title = self._extract_doc_title(doc)
            normalized_title = self._normalize_title(title)

            if normalized_title not in seen_titles:
                score = self._score_document(doc, entities, claim_keywords, idx)
                seen_titles[normalized_title] = True
                unique_docs.append((doc, score))

        # Sort by score (descending) and return top 21
        unique_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in unique_docs[:21]]

    def _select_top_entities(self, entities, top_n=3):
        """Select top N most specific entities, prioritizing uncommon names."""
        if not entities:
            return []

        # Prioritize entities that are likely proper nouns or specific terms
        # Heuristic: longer entities, entities with capital letters, entities with numbers
        scored_entities = []
        for entity in entities:
            score = 0
            # Longer entities are often more specific
            score += len(entity)
            # Entities with capitals are likely proper nouns
            score += sum(1 for c in entity if c.isupper()) * 2
            # Entities with numbers (dates, years) are specific
            score += sum(1 for c in entity if c.isdigit()) * 3
            scored_entities.append((entity, score))

        # Sort by score and return top_n
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, score in scored_entities[:top_n]]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract entities from the claim
            entity_result = self.entity_extractor(claim=claim)
            entities = entity_result.entities if hasattr(entity_result, 'entities') else []

            # Step 2: Generate 1 contextual query
            contextual_result = self.contextual_query_generator(claim=claim)
            contextual_query = contextual_result.query

            # Step 3: Retrieve k=10 documents for contextual query
            all_docs = self.retrieve_k10(contextual_query).passages

            # Step 4: Select top 2-3 most specific entities
            top_entities = self._select_top_entities(entities, top_n=2)

            # Step 5: For each top entity, generate a direct query and retrieve k=10 docs
            # Limit to max 2 entity queries (total 3 queries with contextual)
            for entity in top_entities[:2]:  # Ensure max 3 queries total
                # Use the entity directly as the query
                entity_docs = self.retrieve_k10(entity).passages
                all_docs.extend(entity_docs)

            # Step 6: Deduplicate and rerank
            final_docs = self._deduplicate_and_rerank(all_docs, entities, claim)

            return dspy.Prediction(retrieved_docs=final_docs)
