import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10  # Increased from 7 to 10 per hop (30 total)
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.rerank = dspy.ChainOfThought("claim, passages -> ranked_passages")

    def _normalize_title(self, title):
        """Normalize title for deduplication by removing extra whitespace and converting to lowercase."""
        if not title:
            return ""
        return " ".join(title.lower().strip().split())

    def _deduplicate_documents(self, documents):
        """Remove duplicate documents based on normalized title matching."""
        seen_titles = set()
        unique_docs = []

        for doc in documents:
            # Extract title - assumes doc is a string with format "Title | Content" or similar
            # Adjust parsing logic based on actual document format
            if isinstance(doc, str):
                # Try to extract title (first line or before separator)
                title = doc.split('\n')[0].split('|')[0].strip()
            else:
                # If doc is an object with a title attribute
                title = getattr(doc, 'title', str(doc))

            normalized_title = self._normalize_title(title)

            if normalized_title and normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

        return unique_docs

    def forward(self, claim):
        # HOP 1 - Retrieve 10 documents
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2 - Retrieve 10 documents
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Retrieve 10 documents (30 total)
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all documents from 3 hops (up to 30 documents)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # DEDUPLICATION - Remove documents with identical titles
        deduplicated_docs = self._deduplicate_documents(all_docs)

        # RERANKING - Score and rerank by relevance to claim
        if len(deduplicated_docs) > 0:
            # Pass deduplicated documents to reranker
            reranked_result = self.rerank(claim=claim, passages=deduplicated_docs)

            # Extract ranked passages from the result
            # The ChainOfThought will return ranked_passages
            ranked_docs = reranked_result.ranked_passages

            # If ranked_passages is a string, parse it back to list
            if isinstance(ranked_docs, str):
                # Assume the reranker returns the passages in order
                # For now, use the deduplicated docs as-is if parsing fails
                ranked_docs = deduplicated_docs
            elif not isinstance(ranked_docs, list):
                ranked_docs = deduplicated_docs
        else:
            ranked_docs = deduplicated_docs

        # Return top 21 unique documents after reranking
        final_docs = ranked_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)
