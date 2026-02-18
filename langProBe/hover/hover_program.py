import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        # Multi-query generators for diversified retrieval (Hop 2 and 3)
        self.create_queries_hop2 = dspy.ChainOfThought("claim,summary_1->query", n=2)
        self.create_queries_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query", n=2)

        # Separate retrievers for different hop strategies
        self.retrieve_k_hop1 = dspy.Retrieve(k=7)  # Hop 1: single query, 7 docs
        self.retrieve_k_multi = dspy.Retrieve(k=5)  # Hops 2-3: multi-query, 5 docs per query

        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _extract_title(self, doc: str) -> str:
        """Extract document title from 'title | content' format.

        Args:
            doc: Document string in format "title | content"

        Returns:
            The document title
        """
        return doc.split(" | ")[0]

    def _retrieve_diversified(
        self,
        query_generator,
        seen_titles: set,
        **kwargs
    ) -> list[str]:
        """Generate multiple diverse queries, retrieve documents, deduplicate,
        and return top 7 unique documents.

        This method:
        1. Generates n diverse query perspectives using the query_generator
        2. Retrieves k documents for each query
        3. Deduplicates based on document titles (both within hop and across hops)
        4. Returns up to 7 unique documents in relevance order

        Args:
            query_generator: ChainOfThought module with n>1 configured
            seen_titles: Set of document titles already retrieved in previous hops
            **kwargs: Arguments to pass to query generator (claim, summary_1, etc.)

        Returns:
            List of up to 7 unique document strings in format "title | content"
        """
        # Generate multiple query perspectives (e.g., n=2 queries)
        prediction = query_generator(**kwargs)
        queries = prediction.completions.query  # Access list of n query strings

        # Retrieve documents for each query
        all_docs = []
        for query in queries:
            docs = self.retrieve_k_multi(query).passages
            all_docs.extend(docs)

        # Deduplicate: prioritize unseen documents to maximize information diversity
        unique_docs = []
        for doc in all_docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                unique_docs.append(doc)
                seen_titles.add(title)  # Track for cross-hop deduplication

        # Return top 7 documents (preserving retrieval order = relevance order)
        return unique_docs[:7]

    def forward(self, claim):
        # Track seen document titles across all hops for deduplication
        seen_titles = set()

        # HOP 1: Single query, 7 documents (unchanged)
        hop1_docs = self.retrieve_k_hop1(claim).passages
        for doc in hop1_docs:
            seen_titles.add(self._extract_title(doc))

        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2: Diversified retrieval (2 queries, 5 docs each, deduplicated to 7)
        hop2_docs = self._retrieve_diversified(
            query_generator=self.create_queries_hop2,
            seen_titles=seen_titles,
            claim=claim,
            summary_1=summary_1
        )

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Diversified retrieval (2 queries, 5 docs each, deduplicated to 7)
        hop3_docs = self._retrieve_diversified(
            query_generator=self.create_queries_hop3,
            seen_titles=seen_titles,
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2
        )

        # Return all documents (exactly 21: 7+7+7)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
