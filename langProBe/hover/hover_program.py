import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapFillingQuery(dspy.Signature):
    """You are given a claim, a list of Wikipedia article titles already retrieved, and a summary
    of what has been found so far. Compare the entities/topics explicitly mentioned in the claim
    against the retrieved titles. Generate a targeted search query for the key entity or topic from
    the claim that is NOT yet represented in the retrieved documents. Prefer exact entity names."""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(
        desc="semicolon-separated titles of Wikipedia documents already retrieved in earlier hops"
    )
    summary_2: str = dspy.InputField(
        desc="summary of the context and documents found in hops 1 and 2"
    )
    query: str = dspy.OutputField(
        desc="targeted search query for the missing entity/topic from the claim not yet found"
    )


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
        self.create_query_hop3 = dspy.ChainOfThought(GapFillingQuery)
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

        # HOP 3: gap-filling query — compare claim entities against already-retrieved titles
        # Collect titles from hops 1+2 for explicit gap analysis
        retrieved_titles = "; ".join(
            self._doc_title(d) for d in (hop1_docs[:6] + hop2_docs[:6])
        )

        hop3_query = self.create_query_hop3(
            claim=claim,
            retrieved_titles=retrieved_titles,
            summary_2=summary_2,
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
