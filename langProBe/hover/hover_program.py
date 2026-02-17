import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateMultipleQueries(dspy.Signature):
    """Generate multiple targeted search queries to retrieve relevant documents for a claim.
    Focus on different entities, relationships, or aspects mentioned in the claim."""

    claim = dspy.InputField()
    previous_queries = dspy.InputField(desc="list of queries already generated in previous hops")
    queries = dspy.OutputField(desc="2-3 targeted search queries focusing on different entities, relationships, or aspects mentioned in the claim")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.generate_queries = dspy.ChainOfThought(GenerateMultipleQueries)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # HOP 1: Generate 3 diverse queries from the claim, retrieve k=7 for the first query only
        hop1_result = self.generate_queries(claim=claim, previous_queries=[])
        hop1_queries = self._parse_queries(hop1_result.queries)

        # Use only the first query for retrieval in hop 1
        hop1_docs = self.retrieve_k(hop1_queries[0]).passages if hop1_queries else self.retrieve_k(claim).passages

        # Extract key entities/titles from hop 1 retrieved doc titles for context
        hop1_context = self._extract_doc_titles(hop1_docs)

        # HOP 2: Generate 2-3 new queries considering hop 1 results
        hop2_claim_with_context = f"{claim}\n\nRetrieved documents mention: {hop1_context}"
        hop2_result = self.generate_queries(
            claim=hop2_claim_with_context,
            previous_queries=hop1_queries
        )
        hop2_queries = self._parse_queries(hop2_result.queries)

        # Use the best (first) query for retrieval in hop 2
        hop2_docs = self.retrieve_k(hop2_queries[0]).passages if hop2_queries else []

        # Extract key entities/titles from hop 2 results
        hop2_context = self._extract_doc_titles(hop2_docs)

        # HOP 3: Generate 2-3 queries from remaining information gaps
        hop3_claim_with_context = f"{claim}\n\nAlready retrieved documents about: {hop1_context}, {hop2_context}\n\nFocus on remaining information gaps."
        hop3_result = self.generate_queries(
            claim=hop3_claim_with_context,
            previous_queries=hop1_queries + hop2_queries
        )
        hop3_queries = self._parse_queries(hop3_result.queries)

        # Use the best (first) query for retrieval in hop 3
        hop3_docs = self.retrieve_k(hop3_queries[0]).passages if hop3_queries else []

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)

    def _parse_queries(self, queries_output):
        """Parse the queries output into a list of query strings."""
        if isinstance(queries_output, list):
            return queries_output[:3]  # Take at most 3 queries

        # If it's a string, try to split by newlines or common delimiters
        if isinstance(queries_output, str):
            # Try different parsing strategies
            lines = queries_output.strip().split('\n')
            parsed = []
            for line in lines:
                # Remove common prefixes like "1.", "- ", "• ", etc.
                cleaned = line.strip().lstrip('0123456789.-•* ').strip()
                if cleaned and len(cleaned) > 3:  # Only add non-empty, meaningful queries
                    parsed.append(cleaned)
            return parsed[:3] if parsed else [queries_output]  # Return at most 3 queries

        return [str(queries_output)]

    def _extract_doc_titles(self, docs):
        """Extract key information from document titles/snippets."""
        if not docs:
            return ""

        # Take first few words or key phrases from each doc (typically titles come first)
        titles = []
        for doc in docs[:5]:  # Look at top 5 docs
            # Extract first sentence or up to 50 chars as title/key info
            text = doc if isinstance(doc, str) else str(doc)
            snippet = text.split('.')[0][:50]
            if snippet:
                titles.append(snippet)

        return '; '.join(titles)
