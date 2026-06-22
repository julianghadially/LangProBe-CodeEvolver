import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractNextQuery(dspy.Signature):
    """Multi-hop Wikipedia retrieval. The claim requires finding ~3 Wikipedia articles.
    Identify the single most important Wikipedia article NOT yet in retrieved_titles.

    Follow these steps IN ORDER:

    STEP 1 — CLAIM SCAN: Read the claim carefully. Find any specific named entity
    (person, film, song, TV show, book, company, place, tournament, season, etc.)
    that is explicitly mentioned in the claim text AND does NOT appear in retrieved_titles.
    If such an entity exists, output a short search query for the most important one.

    STEP 2 — PASSAGE SCAN (only use this if all claim entities are already covered by
    retrieved_titles): Read the retrieved passages and find a named entity that:
    (a) is named explicitly in the passages, (b) is directly relevant to verifying the
    claim, and (c) does NOT appear in retrieved_titles.
    Output a short search query for it.

    Output: Just the entity name or title as the query (e.g., 'Elena Shaddow',
    '20th Century Fox', 'Jack Nicholson', '2004-05 Memphis Grizzlies season').
    Do NOT output a sentence or question."""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="retrieved passages from previous hops")
    retrieved_titles: str = dspy.InputField(
        desc="Wikipedia article titles already retrieved — do NOT query for these"
    )
    query: str = dspy.OutputField(
        desc="search query for the most important missing Wikipedia article"
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

        # HOP 2: extract the most important named entity from hop 1 passages
        hop2_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(hop1_docs[:5]),
            retrieved_titles=hop1_titles,
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Combined titles for hop 3
        all_titles_12 = (
            hop1_titles
            + "; "
            + "; ".join(self._doc_title(d) for d in hop2_docs[:10])
        )

        # HOP 3: extract next entity from combined hop 1+2 passages
        hop3_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(hop1_docs[:3] + hop2_docs[:3]),
            retrieved_titles=all_titles_12,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combined titles for hop 4
        all_titles_123 = (
            all_titles_12
            + "; "
            + "; ".join(self._doc_title(d) for d in hop3_docs[:10])
        )

        # HOP 4: extract next entity using all three hops' passages as context.
        hop4_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(hop1_docs[:2] + hop2_docs[:2] + hop3_docs[:2]),
            retrieved_titles=all_titles_123,
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # Combined titles for hop 5
        all_titles_1234 = (
            all_titles_123
            + "; "
            + "; ".join(self._doc_title(d) for d in hop4_docs[:10])
        )

        # HOP 5: final extraction using targeted hops 2-4 passages (most entity-specific).
        # Each additional search beyond 2 costs 0.002; finding 1 more article is worth it.
        hop5_query = self.extract_query(
            claim=claim,
            passages="\n\n".join(hop2_docs[:2] + hop3_docs[:2] + hop4_docs[:2]),
            retrieved_titles=all_titles_1234,
        ).query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # Interleaved round-robin merge with deduplication across all 5 hops.
        seen_titles: set = set()
        final_docs: list = []
        for i in range(self.k):
            for hop_docs in (hop1_docs, hop2_docs, hop3_docs, hop4_docs, hop5_docs):
                if i < len(hop_docs) and len(final_docs) < 21:
                    title = self._doc_title(hop_docs[i])
                    if title not in seen_titles:
                        seen_titles.add(title)
                        final_docs.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=final_docs[:21])
