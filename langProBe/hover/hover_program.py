import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtractionSignature(dspy.Signature):
    """Extract 2-4 specific named entities from a claim.
    Focus on concrete names of people, places, organizations, events, or titles that can be directly searched.
    Avoid abstract concepts or relationships - extract only specific named entities."""

    claim = dspy.InputField(desc="The factual claim to analyze")
    entities = dspy.OutputField(desc="List of 2-4 specific named entities (people, places, organizations, event names, titles). Each entity should be a concrete name, not a description or relationship.")


class EntityQueryGeneratorSignature(dspy.Signature):
    """Generate a search query using specific entity names.
    Create a query that directly includes the entity names to retrieve Wikipedia articles about those entities."""

    claim = dspy.InputField(desc="The original claim being verified")
    target_entities = dspy.InputField(desc="List of specific entity names to focus the search on")
    retrieved_titles = dspy.InputField(desc="Titles of documents already retrieved (to understand what's covered)")

    query = dspy.OutputField(desc="A search query that directly includes the target entity names to retrieve relevant Wikipedia articles")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop retrieval system with entity-focused query expansion.
    Extracts named entities from claims and generates targeted queries using
    specific entity names to retrieve relevant Wikipedia articles.

    EVALUATION
    - Returns exactly 21 documents (7 per hop × 3 hops)
    - Uses entity name matching reranker to ensure precision
    - Retrieves 10 docs per hop, reranks to top 7 based on entity matching'''

    def __init__(self):
        super().__init__()
        self.k = 10  # Retrieve 10 docs per hop
        self.final_k = 7  # Return top 7 after reranking
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Entity-focused modules
        self.entity_extractor = dspy.ChainOfThought(EntityExtractionSignature)
        self.query_generator_hop2 = dspy.ChainOfThought(EntityQueryGeneratorSignature)
        self.query_generator_hop3 = dspy.ChainOfThought(EntityQueryGeneratorSignature)

    def _extract_titles(self, passages: list[str]) -> list[str]:
        """Extract document titles from passages in 'title | content' format"""
        return [passage.split(" | ")[0] for passage in passages]

    def _rerank_by_entity_match(self, passages: list[str], entities: list[str]) -> list[str]:
        """Rerank passages by entity name matching in titles.
        Returns top 7 passages with highest entity match scores."""

        def score_passage(passage: str) -> float:
            """Score a passage based on how many entity names appear in its title"""
            title = passage.split(" | ")[0].lower()
            score = 0
            for entity in entities:
                entity_lower = entity.lower()
                # Exact match or entity appears as substring in title
                if entity_lower in title:
                    score += 2  # Higher weight for matches
                # Partial word matching
                entity_words = entity_lower.split()
                for word in entity_words:
                    if len(word) > 3 and word in title:  # Avoid short common words
                        score += 0.5
            return score

        # Score all passages
        scored_passages = [(passage, score_passage(passage)) for passage in passages]
        # Sort by score (descending), then return top 7
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        return [passage for passage, score in scored_passages[:self.final_k]]

    def _identify_uncovered_entities(self, entities: list[str], retrieved_titles: list[str]) -> list[str]:
        """Identify entities not strongly represented in retrieved document titles"""
        uncovered = []
        retrieved_titles_lower = [title.lower() for title in retrieved_titles]

        for entity in entities:
            entity_lower = entity.lower()
            # Check if entity appears in any retrieved title
            found = False
            for title in retrieved_titles_lower:
                if entity_lower in title:
                    found = True
                    break
            if not found:
                uncovered.append(entity)

        return uncovered

    def forward(self, claim):
        # INITIALIZATION: Extract named entities
        extraction_output = self.entity_extractor(claim=claim)
        entities = extraction_output.entities
        # Ensure entities is a list
        if isinstance(entities, str):
            entities = [e.strip() for e in entities.split(',')]

        all_retrieved_titles = []

        # HOP 1: Direct claim-based retrieval (k=10)
        hop1_docs_raw = self.retrieve_k(claim).passages
        hop1_docs = self._rerank_by_entity_match(hop1_docs_raw, entities)
        hop1_titles = self._extract_titles(hop1_docs)
        all_retrieved_titles.extend(hop1_titles)

        # Identify uncovered entities after Hop 1
        uncovered_entities_hop1 = self._identify_uncovered_entities(entities, hop1_titles)

        # HOP 2: Query using first 1-2 uncovered entities
        if uncovered_entities_hop1:
            target_entities_hop2 = uncovered_entities_hop1[:2]
            hop2_query_output = self.query_generator_hop2(
                claim=claim,
                target_entities=target_entities_hop2,
                retrieved_titles=all_retrieved_titles
            )

            hop2_docs_raw = self.retrieve_k(hop2_query_output.query).passages
            hop2_docs = self._rerank_by_entity_match(hop2_docs_raw, target_entities_hop2)
            hop2_titles = self._extract_titles(hop2_docs)
            all_retrieved_titles.extend(hop2_titles)
        else:
            # Fallback: use original entities
            hop2_query_output = self.query_generator_hop2(
                claim=claim,
                target_entities=entities[:2],
                retrieved_titles=all_retrieved_titles
            )
            hop2_docs_raw = self.retrieve_k(hop2_query_output.query).passages
            hop2_docs = self._rerank_by_entity_match(hop2_docs_raw, entities)
            hop2_titles = self._extract_titles(hop2_docs)
            all_retrieved_titles.extend(hop2_titles)

        # Identify remaining uncovered entities after Hop 2
        uncovered_entities_hop2 = self._identify_uncovered_entities(entities, all_retrieved_titles)

        # HOP 3: Query using remaining uncovered entities
        if uncovered_entities_hop2:
            target_entities_hop3 = uncovered_entities_hop2
            hop3_query_output = self.query_generator_hop3(
                claim=claim,
                target_entities=target_entities_hop3,
                retrieved_titles=all_retrieved_titles
            )

            hop3_docs_raw = self.retrieve_k(hop3_query_output.query).passages
            hop3_docs = self._rerank_by_entity_match(hop3_docs_raw, target_entities_hop3)
        else:
            # Fallback: use all original entities
            hop3_query_output = self.query_generator_hop3(
                claim=claim,
                target_entities=entities,
                retrieved_titles=all_retrieved_titles
            )
            hop3_docs_raw = self.retrieve_k(hop3_query_output.query).passages
            hop3_docs = self._rerank_by_entity_match(hop3_docs_raw, entities)

        # Return exactly 21 documents (7 per hop × 3 hops)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
