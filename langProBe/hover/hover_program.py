import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ExtractPassageEntity(dspy.Signature):
    """Multi-hop Wikipedia retrieval. Examine the retrieved passages and identify the single most
    important named entity (person, film, book, album, TV show, organization, place, etc.) that:
    1. Appears explicitly by name in the retrieved passages
    2. Is directly relevant to verifying the claim
    3. Does NOT already have a Wikipedia article in retrieved_titles

    This entity is typically one that is not named in the claim itself but is found inside the
    retrieved passages (e.g., 'Jack Nicholson' named in the Hoffa article, not in the claim).

    Output: A short, precise search query — just the entity name (e.g., 'Jack Nicholson',
    'Don Giovanni'). NOT a sentence or question."""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="top retrieved passages from the previous hop")
    retrieved_titles: str = dspy.InputField(
        desc="Wikipedia article titles already retrieved, semicolon-separated"
    )
    query: str = dspy.OutputField(
        desc="search query for the most important entity found in passages but NOT yet retrieved"
    )


class FindMissingEntity(dspy.Signature):
    """Multi-hop Wikipedia retrieval gap analysis. The claim requires 3 Wikipedia articles;
    identify the single most important article still missing.

    Check TWO sources in priority order:
    PRIORITY 1 — CLAIM TEXT: Are there specific named entities, titles, or topics explicitly
    stated in the claim that do NOT have a matching article in retrieved_titles? If yes,
    generate a query for the most important one.
    PRIORITY 2 — PASSAGES: If all claim entities are already covered, look in the retrieved
    passages for named entities relevant to the claim that are missing from retrieved_titles.

    Output: A short, precise search query (just the entity name/title, e.g.,
    '2004-05 Memphis Grizzlies season', 'Airlines of Africa', 'Elena Shaddow')."""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="retrieved passages from previous hops (for fallback entities)")
    retrieved_titles: str = dspy.InputField(
        desc="ALL Wikipedia article titles retrieved so far, semicolon-separated"
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
        # Hop 2: extract indirect entities found in retrieved passages
        self.extract_passage_entity = dspy.ChainOfThought(ExtractPassageEntity)
        # Hop 3: gap analysis — claim entities first, then passage entities
        self.find_missing_entity = dspy.ChainOfThought(FindMissingEntity)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def forward(self, claim):
        # HOP 1: retrieve broadly from the raw claim
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = "; ".join(self._doc_title(d) for d in hop1_docs[:12])

        # HOP 2: extract most important INDIRECT entity from hop 1 passages
        # (entity named in passages but NOT in the claim itself)
        hop2_query = self.extract_passage_entity(
            claim=claim,
            passages="\n\n".join(hop1_docs[:5]),
            retrieved_titles=hop1_titles,
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Build combined titles and passages for hop 3
        all_titles = (
            hop1_titles
            + "; "
            + "; ".join(self._doc_title(d) for d in hop2_docs[:12])
        )
        combined_passages = hop1_docs[:3] + hop2_docs[:3]

        # HOP 3: gap-fill — first checks claim entities, then passage entities
        hop3_query = self.find_missing_entity(
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
