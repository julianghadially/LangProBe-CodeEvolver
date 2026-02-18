import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class QueryDecomposition(dspy.Signature):
    """Decompose a claim into 3 distinct sub-queries targeting different entities/aspects.
    Each query should focus on a specific entity, concept, or aspect mentioned in the claim."""

    claim = dspy.InputField()
    query1 = dspy.OutputField(desc="First sub-query targeting a specific entity or aspect")
    query2 = dspy.OutputField(desc="Second sub-query targeting a different entity or aspect")
    query3 = dspy.OutputField(desc="Third sub-query targeting another distinct entity or aspect")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10  # Retrieve 10 documents per query
        self.decompose_claim = dspy.ChainOfThought(QueryDecomposition)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _deduplicate_with_diversity(self, query_results):
        """Deduplicate documents while preserving diversity across query results.

        Args:
            query_results: List of 3 lists, each containing retrieved passages

        Returns:
            List of deduplicated passages maintaining representation from all queries
        """
        seen_titles = {}
        deduplicated = [[], [], []]

        # First pass: keep first occurrence of each unique title
        for query_idx, passages in enumerate(query_results):
            for passage in passages:
                # Extract title (assuming format "title | text" or similar)
                title = passage.split('|')[0].strip() if '|' in passage else passage[:100]

                if title not in seen_titles:
                    seen_titles[title] = query_idx
                    deduplicated[query_idx].append(passage)

        return deduplicated

    def _select_final_documents(self, deduplicated_results):
        """Select final 21 documents by taking top 7 from each query result.

        Args:
            deduplicated_results: List of 3 lists of deduplicated passages

        Returns:
            List of 21 documents (7 from each query to ensure coverage)
        """
        final_docs = []

        # Take top 7 from each query result
        for passages in deduplicated_results:
            final_docs.extend(passages[:7])

        return final_docs

    def forward(self, claim):
        # STEP 1: Decompose claim into 3 distinct sub-queries
        decomposition = self.decompose_claim(claim=claim)
        queries = [decomposition.query1, decomposition.query2, decomposition.query3]

        # STEP 2: Retrieve k=10 documents for each sub-query in parallel (30 total)
        query_results = [
            self.retrieve_k(query).passages
            for query in queries
        ]

        # STEP 3: Diversity-based deduplication
        deduplicated_results = self._deduplicate_with_diversity(query_results)

        # STEP 4: Select final 21 documents (top 7 from each retrieval set)
        final_docs = self._select_final_documents(deduplicated_results)

        return dspy.Prediction(retrieved_docs=final_docs)
