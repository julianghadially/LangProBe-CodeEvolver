import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,retrieved_titles->complementary_query")
        self.create_query_hop2.__doc__ = 'Generate a query that finds documents NOT yet retrieved but needed to verify the claim'
        self.create_query_hop3 = dspy.ChainOfThought("claim,retrieved_titles_hop1,retrieved_titles_hop2->final_query")
        self.create_query_hop3.__doc__ = 'Generate a query that finds documents NOT yet retrieved but needed to verify the claim'
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1
        hop1_result = self.retrieve_k(claim)
        hop1_docs = hop1_result.passages
        hop1_titles = self._extract_titles(hop1_docs)

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim,
            retrieved_titles=hop1_titles
        ).complementary_query
        hop2_result = self.retrieve_k(hop2_query)
        hop2_docs = hop2_result.passages
        hop2_titles = self._extract_titles(hop2_docs)

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim,
            retrieved_titles_hop1=hop1_titles,
            retrieved_titles_hop2=hop2_titles
        ).final_query
        hop3_result = self.retrieve_k(hop3_query)
        hop3_docs = hop3_result.passages

        # Deduplication and ranking
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = self._deduplicate_docs(all_docs)
        ranked_docs = self._rank_by_diversity_and_relevance(unique_docs, claim)

        # Return top 21 unique documents
        final_docs = ranked_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)

    def _extract_titles(self, docs):
        """Extract titles from document passages."""
        titles = []
        for doc in docs:
            # Assuming documents have a title or we can extract the first line
            if isinstance(doc, dict) and 'title' in doc:
                titles.append(doc['title'])
            elif isinstance(doc, str):
                # Extract first line or first N characters as title
                title = doc.split('\n')[0][:100]
                titles.append(title)
        return ', '.join(titles) if titles else ''

    def _deduplicate_docs(self, docs):
        """Remove duplicate documents."""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_str = str(doc)
            if doc_str not in seen:
                seen.add(doc_str)
                unique_docs.append(doc)
        return unique_docs

    def _rank_by_diversity_and_relevance(self, docs, claim):
        """Rank documents by diversity and relevance."""
        # Simple scoring: prioritize documents from later in the list (later hops)
        # as they were explicitly targeting diversity
        # In a real implementation, this could use embeddings or other metrics

        # For now, we reverse to prioritize later-retrieved (more diverse) docs
        # while maintaining some of the original relevance ordering
        scored_docs = []
        for idx, doc in enumerate(docs):
            # Score combines position (diversity proxy) with inverse index (relevance)
            diversity_score = idx / max(len(docs), 1)
            relevance_score = 1.0 - (idx / max(len(docs), 1))
            combined_score = 0.6 * diversity_score + 0.4 * relevance_score
            scored_docs.append((combined_score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]
