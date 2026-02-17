import dspy
import re
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimQueryDiversifier(dspy.Signature):
    """Generate three diverse queries from a claim to retrieve comprehensive information.
    Each query should target a different aspect: entities mentioned, relationships between entities, and verification facts."""

    claim = dspy.InputField(desc="The claim to verify")
    entity_query = dspy.OutputField(desc="Query focused on identifying and retrieving information about specific entities mentioned in the claim")
    relationship_query = dspy.OutputField(desc="Query focused on relationships and connections between entities in the claim")
    verification_query = dspy.OutputField(desc="Query focused on facts and evidence to verify or refute the claim")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 25  # Retrieve 25 docs per query (75 total candidates)
        self.query_diversifier = dspy.ChainOfThought(ClaimQueryDiversifier)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _extract_entities(self, claim):
        """Extract potential entities from claim using simple keyword matching.
        Entities are typically capitalized words or phrases."""
        # Split on common delimiters and extract capitalized sequences
        words = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', claim)
        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', claim)
        entities = set(words + quoted)
        # Convert to lowercase for matching
        return {e.lower() for e in entities}

    def _get_doc_title(self, doc):
        """Extract title from document string (format: 'title | content')"""
        return doc.split(" | ")[0] if " | " in doc else doc.split("\n")[0]

    def _calculate_entity_coverage(self, doc, claim_entities):
        """Count how many distinct claim entities are mentioned in the document."""
        doc_lower = doc.lower()
        return sum(1 for entity in claim_entities if entity in doc_lower)

    def _calculate_title_relevance(self, title, claim):
        """Calculate title relevance using basic text overlap."""
        # Tokenize and convert to lowercase
        title_tokens = set(re.findall(r'\b\w+\b', title.lower()))
        claim_tokens = set(re.findall(r'\b\w+\b', claim.lower()))

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        title_tokens -= stop_words
        claim_tokens -= stop_words

        if not title_tokens or not claim_tokens:
            return 0

        # Calculate Jaccard similarity
        intersection = len(title_tokens & claim_tokens)
        union = len(title_tokens | claim_tokens)
        return intersection / union if union > 0 else 0

    def _rerank_and_deduplicate(self, all_docs, claim, top_k=21):
        """Deduplicate by title and rerank based on entity coverage and title relevance."""
        claim_entities = self._extract_entities(claim)

        # Deduplicate by title while preserving order
        seen_titles = {}
        for doc in all_docs:
            title = self._get_doc_title(doc)
            if title not in seen_titles:
                seen_titles[title] = doc

        unique_docs = list(seen_titles.values())

        # Score each document
        scored_docs = []
        for doc in unique_docs:
            title = self._get_doc_title(doc)
            entity_score = self._calculate_entity_coverage(doc, claim_entities)
            title_score = self._calculate_title_relevance(title, claim)
            # Combined score: entity coverage weighted more heavily
            combined_score = entity_score * 2 + title_score
            scored_docs.append((combined_score, doc))

        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def forward(self, claim):
        # Generate 3 diverse queries in parallel
        query_result = self.query_diversifier(claim=claim)
        entity_query = query_result.entity_query
        relationship_query = query_result.relationship_query
        verification_query = query_result.verification_query

        # Retrieve k=25 documents for each query (75 total candidates)
        entity_docs = self.retrieve_k(entity_query).passages
        relationship_docs = self.retrieve_k(relationship_query).passages
        verification_docs = self.retrieve_k(verification_query).passages

        # Combine all candidates
        all_candidates = entity_docs + relationship_docs + verification_docs

        # Rerank and deduplicate to get top 21 documents
        final_docs = self._rerank_and_deduplicate(all_candidates, claim, top_k=21)

        return dspy.Prediction(retrieved_docs=final_docs)
