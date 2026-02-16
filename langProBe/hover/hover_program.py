import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyMissingInfo(dspy.Signature):
    """Analyze retrieved passages and identify what key entities, facts, or aspects are still missing to verify the claim."""

    claim = dspy.InputField()
    retrieved_passages = dspy.InputField(desc="list of passages retrieved so far")
    missing_aspects = dspy.OutputField(desc="description of what key entities/facts are still needed")


class GenerateTargetedQuery(dspy.Signature):
    """Generate a targeted search query to find the missing information identified in the gap analysis."""

    claim = dspy.InputField()
    missing_aspects = dspy.InputField(desc="description of missing information")
    query = dspy.OutputField(desc="targeted search query to find the missing information")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.identify_missing = dspy.ChainOfThought(IdentifyMissingInfo)
        self.generate_query = dspy.ChainOfThought(GenerateTargetedQuery)

    def _deduplicate_by_title(self, docs):
        """Deduplicate documents by their title, keeping first occurrence."""
        seen_titles = set()
        unique_docs = []
        for doc in docs:
            # Extract title - assuming passages are strings or have a consistent format
            # If passages have a title attribute, use: doc_title = getattr(doc, 'title', doc)
            # For now, treating the passage itself as the identifier
            doc_str = str(doc)
            if doc_str not in seen_titles:
                seen_titles.add(doc_str)
                unique_docs.append(doc)
        return unique_docs

    def forward(self, claim):
        # HOP 1: Retrieve k=7 documents directly from the claim
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Identify missing information from hop1 and generate targeted query
        missing_info_hop2 = self.identify_missing(
            claim=claim,
            retrieved_passages=hop1_docs
        ).missing_aspects

        hop2_query = self.generate_query(
            claim=claim,
            missing_aspects=missing_info_hop2
        ).query

        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Identify remaining gaps from hop1+hop2 and generate final targeted query
        combined_docs_hop1_hop2 = hop1_docs + hop2_docs

        missing_info_hop3 = self.identify_missing(
            claim=claim,
            retrieved_passages=combined_docs_hop1_hop2
        ).missing_aspects

        hop3_query = self.generate_query(
            claim=claim,
            missing_aspects=missing_info_hop3
        ).query

        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all documents and deduplicate by title to ensure 21 unique documents
        all_docs = hop1_docs + hop2_docs + hop3_docs
        unique_docs = self._deduplicate_by_title(all_docs)

        return dspy.Prediction(retrieved_docs=unique_docs)
