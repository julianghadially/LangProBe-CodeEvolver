import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop
from collections import Counter
import math

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityExtractionSignature(dspy.Signature):
    """Extract 4-6 specific named entities (people, places, works, events) from the claim."""

    claim: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(
        desc="4-6 specific named entities like people, places, works, or events mentioned in the claim"
    )


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.entity_extractor = dspy.Predict(EntityExtractionSignature)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Extract entities from the claim
            extraction_result = self.entity_extractor(claim=claim)
            entities = extraction_result.entities

            # Limit to first 3 entities as per constraint (max 3 queries)
            entities = entities[:3]

            # Retrieve k=100 documents per entity query
            all_docs = []
            for entity in entities:
                # Simple query: just the entity name or 2-3 word phrase
                query = entity.strip()
                if query:
                    try:
                        retrieved = self.rm(query, k=100)
                        if hasattr(retrieved, 'passages'):
                            all_docs.extend(retrieved.passages)
                    except:
                        # If retrieval fails, continue with other entities
                        pass

            # Deduplicate documents
            unique_docs = []
            seen_texts = set()
            for doc in all_docs:
                doc_text = doc if isinstance(doc, str) else str(doc)
                if doc_text not in seen_texts:
                    seen_texts.add(doc_text)
                    unique_docs.append(doc)

            # Rerank using lexical overlap scoring
            scored_docs = []
            for doc in unique_docs:
                score = self._compute_lexical_overlap_score(claim, doc)
                scored_docs.append((doc, score))

            # Sort by score descending and select top 21
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_21_docs = [doc for doc, score in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)

    def _compute_lexical_overlap_score(self, claim, doc):
        """Compute lexical overlap score using n-gram matching weighted by term rarity."""
        # Normalize text to lowercase for matching
        claim_lower = claim.lower()
        doc_text = doc if isinstance(doc, str) else str(doc)
        doc_lower = doc_text.lower()

        # Tokenize claim and document
        claim_tokens = claim_lower.split()
        doc_tokens = doc_lower.split()

        # Count term frequencies in document for rarity weighting
        doc_term_freq = Counter(doc_tokens)
        total_terms = len(doc_tokens) if doc_tokens else 1

        score = 0.0

        # Unigrams (single words)
        claim_unigrams = set(claim_tokens)
        for unigram in claim_unigrams:
            if unigram in doc_lower:
                # Weight by inverse frequency (rarer terms get higher weight)
                tf = doc_term_freq.get(unigram, 0)
                if tf > 0:
                    # IDF-like weighting: rare terms (low tf) get higher scores
                    idf_weight = math.log(total_terms / tf) if tf > 0 else 0
                    score += 1.0 * idf_weight

        # Bigrams (2-word phrases)
        claim_bigrams = [' '.join(claim_tokens[i:i+2]) for i in range(len(claim_tokens)-1)]
        for bigram in claim_bigrams:
            if bigram in doc_lower:
                # Bigrams are more specific, give higher weight
                score += 2.0

        # Trigrams (3-word phrases)
        claim_trigrams = [' '.join(claim_tokens[i:i+3]) for i in range(len(claim_tokens)-2)]
        for trigram in claim_trigrams:
            if trigram in doc_lower:
                # Trigrams are most specific, give highest weight
                score += 3.0

        return score
