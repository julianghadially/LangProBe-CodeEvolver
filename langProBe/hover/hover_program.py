import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with adaptive deduplication.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Implements Adaptive Deduplicated Multi-Hop Retrieval
    - Deduplicates documents across hops by tracking document titles
    - Dynamically adjusts retrieval size to maintain target document count
    - Prevents duplicate documents from wasting retrieval slots'''

    MAX_RETRIEVED_DOCS = 21

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _extract_title(self, doc):
        """Extract document title from passage text (text before ' | ')."""
        if isinstance(doc, str):
            return doc.split(" | ")[0] if " | " in doc else doc
        # Handle potential dict-like objects
        text = getattr(doc, 'text', str(doc))
        return text.split(" | ")[0] if " | " in text else text

    def _deduplicate_docs(self, docs, seen_titles):
        """Deduplicate documents by title, returning only new unique documents."""
        unique_docs = []
        for doc in docs:
            title = self._extract_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        return unique_docs

    def _calculate_adaptive_k(self, current_unique_count, hop_number, base_k=7):
        """Calculate adaptive k for next hop to reach MAX_RETRIEVED_DOCS target.

        Args:
            current_unique_count: Number of unique documents collected so far
            hop_number: Current hop number (1-indexed)
            base_k: Base retrieval size

        Returns:
            Adjusted k value for next retrieval
        """
        remaining_docs = self.MAX_RETRIEVED_DOCS - current_unique_count

        # If we're close to target, retrieve exactly what we need
        if remaining_docs <= 0:
            return 0  # Already have enough docs

        # Calculate remaining hops (assuming 3 total hops)
        remaining_hops = 3 - hop_number

        if remaining_hops <= 0:
            return remaining_docs

        # If we're significantly behind target, increase k
        expected_docs_per_hop = remaining_docs / remaining_hops

        # Use at least the expected amount, with a buffer for potential duplicates
        # The buffer accounts for the likelihood of duplicates in subsequent hops
        adaptive_k = max(base_k, int(expected_docs_per_hop * 1.5))

        # Cap at a reasonable maximum to avoid excessive retrieval
        return min(adaptive_k, 15)

    def forward(self, claim):
        # Track unique document titles across all hops
        seen_titles = set()
        all_unique_docs = []

        # HOP 1
        hop1_docs = dspy.Retrieve(k=self.k)(claim).passages
        unique_hop1_docs = self._deduplicate_docs(hop1_docs, seen_titles)
        all_unique_docs.extend(unique_hop1_docs)

        summary_1 = self.summarize1(
            claim=claim, passages=unique_hop1_docs
        ).summary  # Summarize unique docs from hop 1

        # Calculate adaptive k for hop 2
        current_count = len(all_unique_docs)
        adaptive_k2 = self._calculate_adaptive_k(current_count, hop_number=1, base_k=self.k)

        # HOP 2
        if adaptive_k2 > 0 and current_count < self.MAX_RETRIEVED_DOCS:
            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = dspy.Retrieve(k=adaptive_k2)(hop2_query).passages
            unique_hop2_docs = self._deduplicate_docs(hop2_docs, seen_titles)
            all_unique_docs.extend(unique_hop2_docs)

            summary_2 = self.summarize2(
                claim=claim, context=summary_1, passages=unique_hop2_docs
            ).summary
        else:
            summary_2 = summary_1  # Reuse summary if no new docs needed

        # Calculate adaptive k for hop 3
        current_count = len(all_unique_docs)
        adaptive_k3 = self._calculate_adaptive_k(current_count, hop_number=2, base_k=self.k)

        # HOP 3
        if adaptive_k3 > 0 and current_count < self.MAX_RETRIEVED_DOCS:
            hop3_query = self.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query
            hop3_docs = dspy.Retrieve(k=adaptive_k3)(hop3_query).passages
            unique_hop3_docs = self._deduplicate_docs(hop3_docs, seen_titles)
            all_unique_docs.extend(unique_hop3_docs)

        # Return exactly up to MAX_RETRIEVED_DOCS unique documents
        final_docs = all_unique_docs[:self.MAX_RETRIEVED_DOCS]

        return dspy.Prediction(retrieved_docs=final_docs)
