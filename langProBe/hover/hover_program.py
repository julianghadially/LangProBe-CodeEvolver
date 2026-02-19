import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class EntityExtraction(dspy.Signature):
    """Extract 2-4 key named entities or topics from the claim that are critical for verification.
    Focus on people, organizations, locations, events, or specific concepts mentioned."""

    claim: str = dspy.InputField(desc="the claim to analyze")
    entities: list[str] = dspy.OutputField(desc="list of 2-4 key named entities or topics from the claim")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with entity-focused query decomposition.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.entity_extractor = dspy.Predict(EntityExtraction)
        self.max_hops = 3
        self.max_total_searches = 3  # Constraint: max 3 search operations

    def _extract_title(self, doc):
        """Extract document title from document string."""
        return doc.split(" | ")[0] if " | " in doc else doc[:100]

    def _calculate_relevance_score(self, doc, claim_keywords, entities):
        """Calculate relevance score based on keyword/entity presence in title and content."""
        doc_lower = doc.lower()
        title = self._extract_title(doc).lower()

        score = 0.0

        # Check for claim keywords (extracted as lowercase tokens)
        for keyword in claim_keywords:
            if keyword in title:
                score += 3.0  # High weight for title matches
            elif keyword in doc_lower:
                score += 1.0  # Lower weight for content matches

        # Check for entities
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in title:
                score += 5.0  # Very high weight for entity in title
            elif entity_lower in doc_lower:
                score += 2.0  # Higher weight for entity in content

        return score

    def _extract_claim_keywords(self, claim):
        """Extract important keywords from claim (simple tokenization, filter stopwords)."""
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'can', 'may', 'might', 'must', 'of', 'in', 'on', 'at', 'to',
                     'for', 'with', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'between', 'under', 'over', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                     'that', 'this', 'these', 'those', 'and', 'but', 'or', 'if', 'because',
                     'as', 'until', 'while'}

        # Simple word tokenization
        words = claim.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()
        keywords = [w for w in words if len(w) > 2 and w not in stopwords]
        return keywords

    def forward(self, claim):
        # Track unique documents by title
        seen_titles = set()
        all_docs = []  # Store all retrieved documents before ranking
        search_count = 0  # Track number of search operations

        # Extract claim keywords for relevance scoring
        claim_keywords = self._extract_claim_keywords(claim)

        # HOP 1: Initial broad retrieval for the original claim
        hop1_retrieval = dspy.Retrieve(k=10)(claim)
        search_count += 1
        hop1_docs = hop1_retrieval.passages

        # Store unique documents from hop 1
        for doc in hop1_docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                all_docs.append(doc)

        # HOP 2: Entity-focused query decomposition
        # Extract 2-4 key entities from the claim
        entity_result = self.entity_extractor(claim=claim)
        entities = entity_result.entities if hasattr(entity_result, 'entities') else []

        # Ensure we have entities; fallback to claim keywords if extraction fails
        if not entities or len(entities) == 0:
            entities = claim_keywords[:4]  # Use top keywords as fallback

        # Limit to 2-4 entities
        entities = entities[:4] if len(entities) > 4 else entities
        entities = entities[:2] if len(entities) < 2 else entities

        # Generate 1-2 specific entity queries (staying within 3 searches total)
        # We already used 1 search, so we can do at most 2 more
        num_entity_queries = min(2, len(entities), self.max_total_searches - search_count)

        for i in range(num_entity_queries):
            if search_count >= self.max_total_searches:
                break

            entity = entities[i]
            # Create focused query combining claim context with specific entity
            entity_query = f"{entity} {claim}"

            # Retrieve k=8-10 docs per entity query
            k_entity = 10 if i == 0 else 8  # First entity gets k=10, second gets k=8
            hop2_retrieval = dspy.Retrieve(k=k_entity)(entity_query)
            search_count += 1
            hop2_docs = hop2_retrieval.passages

            # Store unique documents from hop 2
            for doc in hop2_docs:
                title = self._extract_title(doc)
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs.append(doc)

        # HOP 3: Gap-filling retrieval if needed
        # Only perform if we haven't reached search limit and have fewer than 50 docs
        if search_count < self.max_total_searches and len(all_docs) < 50:
            # Synthesize a gap-filling query focusing on missing information
            # Use remaining entities or a comprehensive query
            if len(entities) > num_entity_queries:
                # Use next entity
                gap_entity = entities[num_entity_queries]
                gap_query = f"{gap_entity} {claim}"
            else:
                # Combine multiple entities for comprehensive coverage
                combined_entities = " ".join(entities[:3])
                gap_query = f"{combined_entities} {claim}"

            hop3_retrieval = dspy.Retrieve(k=15)(gap_query)
            search_count += 1
            hop3_docs = hop3_retrieval.passages

            # Store unique documents from hop 3
            for doc in hop3_docs:
                title = self._extract_title(doc)
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs.append(doc)

        # Ensure we have up to 50 total documents before reranking
        # If we have fewer, pad with additional retrieval
        if len(all_docs) < 50 and search_count < self.max_total_searches:
            padding_needed = min(50 - len(all_docs), 30)
            padding_retrieval = dspy.Retrieve(k=padding_needed)(claim)
            for doc in padding_retrieval.passages:
                title = self._extract_title(doc)
                if title not in seen_titles and len(all_docs) < 50:
                    seen_titles.add(title)
                    all_docs.append(doc)

        # Relevance-based reranking: score each document
        docs_with_scores = []
        for doc in all_docs:
            score = self._calculate_relevance_score(doc, claim_keywords, entities)
            docs_with_scores.append((doc, score))

        # Sort by score (descending) and take top 21
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in docs_with_scores[:21]]

        return dspy.Prediction(retrieved_docs=final_docs)
