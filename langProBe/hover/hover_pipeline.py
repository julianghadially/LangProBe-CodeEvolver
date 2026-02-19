import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from typing import List
import re

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class EntityQueryGenerator(dspy.Signature):
    """Generate entity-focused queries from a claim for document retrieval."""
    claim: str = dspy.InputField()
    queries: List[str] = dspy.OutputField(desc="2-3 entity-focused search queries targeting different aspects of the claim")


class GapAnalysis(dspy.Signature):
    """Analyze retrieved documents to identify missing entities or concepts mentioned in the claim."""
    claim: str = dspy.InputField()
    retrieved_passages: str = dspy.InputField(desc="Summary of documents retrieved so far")
    missing_concepts: List[str] = dspy.OutputField(desc="1-2 gap-filling queries targeting missing entities/concepts from the claim")


class KeyTermExtractor(dspy.Signature):
    """Extract key terms that are still missing from retrieved documents."""
    claim: str = dspy.InputField()
    retrieved_passages: str = dspy.InputField(desc="Summary of all documents retrieved")
    key_terms: List[str] = dspy.OutputField(desc="Key terms or phrases still missing that need targeted retrieval")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize query generators for each hop
        self.entity_query_gen = dspy.Predict(EntityQueryGenerator)
        self.gap_analyzer = dspy.ChainOfThought(GapAnalysis)
        self.key_term_extractor = dspy.ChainOfThought(KeyTermExtractor)

    def _deduplicate_docs(self, docs: List[str]) -> List[str]:
        """Deduplicate documents by title (first part before ' | ')."""
        seen_titles = set()
        unique_docs = []
        for doc in docs:
            title = doc.split(" | ")[0]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        return unique_docs

    def _extract_entities_from_claim(self, claim: str) -> List[str]:
        """Extract key entities and concepts from the claim for coverage scoring."""
        # Extract capitalized words and phrases (potential entities)
        entities = []

        # Find capitalized words (potential proper nouns)
        words = claim.split()
        for word in words:
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and len(clean_word) > 2 and clean_word[0].isupper():
                entities.append(clean_word.lower())

        # Also include important common words (excluding stop words)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'in', 'a', 'an', 'and', 'or', 'but', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'are', 'to', 'of', 'for', 'with', 'by', 'from', 'as', 'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she', 'they', 'their'}

        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if clean_word and len(clean_word) > 3 and clean_word not in stop_words:
                entities.append(clean_word)

        return list(set(entities))  # Remove duplicates

    def _calculate_coverage_score(self, doc: str, entities: List[str]) -> int:
        """Calculate coverage score by counting how many claim entities/concepts the document mentions."""
        doc_lower = doc.lower()
        score = 0
        for entity in entities:
            if entity in doc_lower:
                score += 1
        return score

    def _rerank_by_coverage(self, docs: List[str], claim: str, top_k: int = 21) -> List[str]:
        """Rerank documents by coverage score and return top k."""
        entities = self._extract_entities_from_claim(claim)

        # Calculate scores for each document
        doc_scores = []
        for doc in docs:
            score = self._calculate_coverage_score(doc, entities)
            doc_scores.append((doc, score))

        # Sort by score (descending) and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:top_k]]

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            all_docs = []

            # ===== HOP 1: Generate 2-3 entity-focused queries and retrieve k=50 per query =====
            hop1_queries_result = self.entity_query_gen(claim=claim)
            hop1_queries = hop1_queries_result.queries

            # Ensure we have 2-3 queries
            if isinstance(hop1_queries, str):
                # If it's a single string, try to split it
                hop1_queries = [q.strip() for q in hop1_queries.split('\n') if q.strip()]
            hop1_queries = hop1_queries[:3]  # Limit to 3 queries

            # Retrieve k=50 docs per query
            retrieve_hop1 = dspy.Retrieve(k=50)
            for query in hop1_queries:
                if query:  # Skip empty queries
                    docs = retrieve_hop1(query).passages
                    all_docs.extend(docs)

            # Deduplicate after hop 1
            all_docs = self._deduplicate_docs(all_docs)

            # ===== HOP 2: Analyze gaps and retrieve k=30 per gap-filling query =====
            # Create summary of retrieved docs for gap analysis
            retrieved_summary = "\n".join([doc[:200] for doc in all_docs[:10]])  # Sample first 10 docs

            gap_result = self.gap_analyzer(claim=claim, retrieved_passages=retrieved_summary)
            gap_queries = gap_result.missing_concepts

            # Ensure we have 1-2 gap queries
            if isinstance(gap_queries, str):
                gap_queries = [q.strip() for q in gap_queries.split('\n') if q.strip()]
            gap_queries = gap_queries[:2]  # Limit to 2 queries

            # Retrieve k=30 docs per gap query
            retrieve_hop2 = dspy.Retrieve(k=30)
            for query in gap_queries:
                if query:  # Skip empty queries
                    docs = retrieve_hop2(query).passages
                    all_docs.extend(docs)

            # Deduplicate after hop 2
            all_docs = self._deduplicate_docs(all_docs)

            # ===== HOP 3: Extract missing key terms and do final retrieval with k=20 =====
            # Update retrieved summary with more docs
            retrieved_summary = "\n".join([doc[:200] for doc in all_docs[:15]])  # Sample first 15 docs

            key_terms_result = self.key_term_extractor(claim=claim, retrieved_passages=retrieved_summary)
            key_terms = key_terms_result.key_terms

            # Ensure we have key terms as a list
            if isinstance(key_terms, str):
                key_terms = [t.strip() for t in key_terms.split('\n') if t.strip()]

            # Retrieve k=20 docs for each key term (or combined query)
            retrieve_hop3 = dspy.Retrieve(k=20)
            if key_terms:
                # Combine key terms into a focused query
                combined_query = " ".join(key_terms[:3])  # Use top 3 key terms
                if combined_query:
                    docs = retrieve_hop3(combined_query).passages
                    all_docs.extend(docs)

            # Final deduplication
            all_docs = self._deduplicate_docs(all_docs)

            # ===== Coverage-based reranking: Select top 21 unique docs by coverage score =====
            final_docs = self._rerank_by_coverage(all_docs, claim, top_k=21)

            return dspy.Prediction(retrieved_docs=final_docs)
