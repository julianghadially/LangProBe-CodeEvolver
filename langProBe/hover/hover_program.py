import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7

        # Entity extraction
        self.entity_extractor = dspy.Predict("claim->entities")

        # Query generation (using context instead of summaries)
        self.create_query_hop2 = dspy.Predict("claim,context->query")
        self.create_query_hop3 = dspy.Predict("claim,context->query")

        # Retrieval
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _parallel_retrieve(self, queries):
        """
        Retrieve k documents for each query in parallel.

        Args:
            queries: List of search query strings

        Returns:
            List of all retrieved documents (may contain duplicates)
        """
        all_docs = []
        for query in queries:
            docs = self.retrieve_k(query).passages
            all_docs.extend(docs)
        return all_docs

    def _get_document_titles(self, docs):
        """
        Extract titles from document passages for context.
        Assumes documents start with a title line or uses first N chars.

        Args:
            docs: List of document passages

        Returns:
            String of comma-separated titles
        """
        titles = []
        for doc in docs:
            # Extract title (first line or first 50 chars)
            title = doc.split('\n')[0][:50] if '\n' in doc else doc[:50]
            titles.append(title.strip())
        return ", ".join(titles)

    def forward(self, claim):
        # Extract 2-4 entities from the claim
        entities_str = self.entity_extractor(claim=claim).entities
        entities = [e.strip() for e in entities_str.split(',') if e.strip()][:4]

        # Fallback: if fewer than 2 entities, pad with claim
        if len(entities) < 2:
            entities = entities + [claim] * (2 - len(entities))
        entities = entities[:4]  # Cap at 4

        # HOP 1: Parallel retrieval with claim + entities
        hop1_queries = [claim] + entities
        hop1_docs = self._parallel_retrieve(hop1_queries)
        hop1_titles = self._get_document_titles(hop1_docs)

        # HOP 2: Generate query using claim + hop1 titles, then parallel retrieval
        hop2_query = self.create_query_hop2(claim=claim, context=hop1_titles).query
        hop2_queries = [hop2_query] + entities
        hop2_docs = self._parallel_retrieve(hop2_queries)
        hop2_titles = self._get_document_titles(hop2_docs)

        # HOP 3: Generate query using claim + hop1+hop2 titles, then parallel retrieval
        combined_context = hop1_titles + " | " + hop2_titles
        hop3_query = self.create_query_hop3(claim=claim, context=combined_context).query
        hop3_queries = [hop3_query] + entities
        hop3_docs = self._parallel_retrieve(hop3_queries)

        # Deduplicate and limit to 21 documents
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = deduplicate(all_docs)[:21]

        return dspy.Prediction(retrieved_docs=unique_docs)
