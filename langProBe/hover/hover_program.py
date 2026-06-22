import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractNextQuery(dspy.Signature):
    """Multi-hop Wikipedia retrieval system. The claim requires finding 3 Wikipedia articles.

    Task: Examine the retrieved passages and identify the single most important named entity
    (person, film, book, album, TV show, organization, place, etc.) that:
    1. Appears explicitly by name in the retrieved passages
    2. Is directly relevant to verifying the claim
    3. Does NOT already have a Wikipedia article in retrieved_titles

    Output: A short, precise search query — typically just the entity name (e.g., 'Jack Nicholson',
    'Don Giovanni', 'Mars Incorporated', 'The Rescuers'). NOT a question or sentence."""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="top retrieved passages from previous search(es)")
    retrieved_titles: str = dspy.InputField(
        desc="Wikipedia article titles already retrieved, semicolon-separated — do NOT generate a query for these"
    )
    query: str = dspy.OutputField(
        desc="search query for the most important missing Wikipedia article — just the entity name/title"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program."""

    def __init__(self):
        super().__init__()
        self.k = 25
        self.extract_query = dspy.ChainOfThought(ExtractNextQuery)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def forward(self, claim):
        # HOP 1: retrieve broadly from the raw claim
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = "; ".join(self._doc_title(d) for d in hop1_docs[:10])

        # HOP 2: extract the most important named entity from hop 1 passages that is
        # not yet retrieved, then search for it
        hop2_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(hop1_docs[:5]),
            retrieved_titles=hop1_titles,
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Build combined context for hop 3
        all_titles = (
            hop1_titles
            + "; "
            + "; ".join(self._doc_title(d) for d in hop2_docs[:10])
        )
        combined_passages = hop1_docs[:3] + hop2_docs[:3]

        # HOP 3: extract next entity from combined hop 1+2 passages
        hop3_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(combined_passages),
            retrieved_titles=all_titles,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Interleaved round-robin merge with deduplication.
        # At each round i, take position i from hop1, then hop2, then hop3
        # (skipping duplicates by title).
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
