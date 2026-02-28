import dspy
import re
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class ClaimDecomposition(dspy.Signature):
    """Extract 2-3 highly specific search phrases from a claim. Focus on:
    (1) proper nouns and named entities
    (2) unique descriptive phrases
    (3) relationship keywords
    Each query should target one specific fact rather than broad context."""

    claim: str = dspy.InputField(desc="the claim to decompose")
    search_phrase_1: str = dspy.OutputField(desc="first specific search phrase focusing on proper nouns/entities")
    search_phrase_2: str = dspy.OutputField(desc="second specific search phrase focusing on unique descriptive phrases")
    search_phrase_3: str = dspy.OutputField(desc="third specific search phrase focusing on relationship keywords")


class DocumentReranker(dspy.Module):
    """Score-based reranking module that computes relevance scores based on
    exact phrase matching and entity overlap with the original claim."""

    def __init__(self):
        super().__init__()

    def _extract_entities(self, text):
        """Extract potential entities and key phrases from text."""
        # Normalize text
        text_lower = text.lower()
        # Extract words (filter out common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'should', 'could', 'may', 'might', 'must', 'can', 'that',
                      'this', 'these', 'those', 'it', 'its', 'which', 'who', 'when', 'where',
                      'why', 'how', 'what', 'all', 'each', 'every', 'both', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                      'same', 'so', 'than', 'too', 'very'}

        words = re.findall(r'\b[a-z]+\b', text_lower)
        entities = [w for w in words if w not in stop_words and len(w) > 2]

        # Extract capitalized phrases (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        return set(entities), set(n.lower() for n in proper_nouns)

    def _compute_score(self, claim, doc_text):
        """Compute relevance score based on exact phrase matching and entity overlap."""
        claim_entities, claim_proper = self._extract_entities(claim)
        doc_entities, doc_proper = self._extract_entities(doc_text)

        # Score components
        # 1. Entity overlap score (Jaccard similarity)
        if claim_entities:
            entity_overlap = len(claim_entities & doc_entities) / len(claim_entities)
        else:
            entity_overlap = 0.0

        # 2. Proper noun overlap (weighted higher)
        if claim_proper:
            proper_overlap = len(claim_proper & doc_proper) / len(claim_proper)
        else:
            proper_overlap = 0.0

        # 3. Exact phrase matching (check if key bigrams/trigrams from claim appear in doc)
        claim_lower = claim.lower()
        doc_lower = doc_text.lower()
        claim_words = claim_lower.split()

        phrase_match_score = 0.0
        phrase_count = 0
        # Check bigrams
        for i in range(len(claim_words) - 1):
            bigram = ' '.join(claim_words[i:i+2])
            if len(bigram) > 5 and bigram in doc_lower:  # Skip very short phrases
                phrase_match_score += 1
            phrase_count += 1

        if phrase_count > 0:
            phrase_match_score /= phrase_count

        # Weighted combination
        total_score = (0.3 * entity_overlap +
                      0.4 * proper_overlap +
                      0.3 * phrase_match_score)

        return total_score

    def rerank(self, claim, documents, top_k=7):
        """Rerank documents based on relevance scores and return top_k."""
        scored_docs = []
        for doc in documents:
            score = self._compute_score(claim, doc)
            scored_docs.append((score, doc))

        # Sort by score (descending) and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using claim decomposition.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    STRATEGY
    - Decomposes claim into 2-3 specific search phrases targeting entities and relationships
    - Retrieves k=25 documents per query (over-retrieval)
    - Applies score-based reranking to select top 7 documents per query
    - Returns exactly 21 total documents (3 queries × 7 docs)'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()

        # New claim decomposition components
        self.claim_decomposer = dspy.Predict(ClaimDecomposition)
        self.retrieve_k25 = dspy.Retrieve(k=25)
        self.reranker = DocumentReranker()

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Decompose claim into 2-3 specific search phrases
            decomposition = self.claim_decomposer(claim=claim)
            search_phrases = [
                decomposition.search_phrase_1,
                decomposition.search_phrase_2,
                decomposition.search_phrase_3
            ]

            # Step 2: Over-retrieve k=25 documents per query
            all_reranked_docs = []
            for phrase in search_phrases:
                # Retrieve 25 documents for this query
                retrieved_docs = self.retrieve_k25(phrase).passages

                # Step 3: Apply score-based reranking to get top 7
                top_7_docs = self.reranker.rerank(claim, retrieved_docs, top_k=7)
                all_reranked_docs.extend(top_7_docs)

            # Return exactly 21 documents (3 queries × 7 docs each)
            return dspy.Prediction(retrieved_docs=all_reranked_docs)
