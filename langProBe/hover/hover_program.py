import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Retrieve 21 candidates per hop (up from 7). Interleaved deduplication
        # below ensures all 3 hops contribute equally to the 21-doc output budget,
        # while tripling coverage per hop.
        self.k = 21
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def forward(self, claim):
        # HOP 1: retrieve broadly from the raw claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs[:7]  # top 7 keeps summary crisp
        ).summary

        # HOP 2: targeted query built from what we found in hop 1
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs[:7]
        ).summary

        # HOP 3: gap-filling query based on both summaries
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Interleaved round-robin merge with deduplication.
        # At each round i, we take position i from hop1, then hop2, then hop3
        # (skipping duplicates by title). This ensures all three hops contribute
        # equally rather than hop1 monopolising the 21-doc budget.
        seen_titles: set = set()
        final_docs: list = []
        for i in range(self.k):
            for hop_docs in (hop1_docs, hop2_docs, hop3_docs):
                if i < len(hop_docs) and len(final_docs) < 21:
                    title = self._doc_title(hop_docs[i])
                    if title not in seen_titles:
                        seen_titles.add(title)
                        final_docs.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=final_docs[:21])
