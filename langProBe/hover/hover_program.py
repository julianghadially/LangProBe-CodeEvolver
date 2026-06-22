import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapFillingEntities(dspy.Signature):
    """Given a claim and the Wikipedia articles already retrieved, identify the two most important
    named entities from the claim that are NOT yet covered by the retrieved documents.
    Return exact entity names as they would appear as Wikipedia article titles.
    Examples: 'Rogue One', 'University of Florida', 'Caroline Wozniacki', 'Mars Incorporated'.
    Focus on specific people, films, places, songs, organizations explicitly mentioned in the claim."""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(
        desc="semicolon-separated titles of Wikipedia documents already retrieved in earlier hops"
    )
    summary: str = dspy.InputField(
        desc="summary of context and key entities found in hops 1 and 2"
    )
    entity1: str = dspy.OutputField(
        desc="most important entity from the claim not yet found in retrieved titles, as exact Wikipedia article title"
    )
    entity2: str = dspy.OutputField(
        desc="second most important entity from the claim not yet found, as exact Wikipedia article title"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # 25 candidates per hop (max allowed). Four hops × 25 = 100 candidates,
        # deduped down to 21. Hops 3 & 4 directly target the two most likely
        # missing entities by exact Wikipedia-title search.
        self.k = 25
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.extract_gap_entities = dspy.ChainOfThought(GapFillingEntities)
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

        # HOPS 3 & 4: extract the two most-missing entities from the claim and
        # search for each one directly by exact Wikipedia-title query.
        # Using top 10 from each of hops 1+2 for a thorough gap analysis.
        retrieved_titles = "; ".join(
            self._doc_title(d) for d in (hop1_docs[:10] + hop2_docs[:10])
        )

        gap = self.extract_gap_entities(
            claim=claim,
            retrieved_titles=retrieved_titles,
            summary=summary_2,
        )
        hop3_docs = self.retrieve_k(gap.entity1).passages
        hop4_docs = self.retrieve_k(gap.entity2).passages

        # Tiered merge with deduplication:
        #   Phase 1 — top 7 unique docs from hop1 (broad claim search, highest precision)
        #   Phase 2 — top 7 unique docs from hop2 not already in hop1
        #   Phase 3 — fill remaining 7 slots via round-robin from entity-targeted hops 3+4
        #
        # This guarantees hops 1 & 2 maintain their full 7-slot quota (same as the
        # original 3-hop baseline), while adding 7 entity-targeted slots from hops 3+4.
        # Strictly better than the 3-hop baseline for "1 doc missing" failures.
        seen_titles: set = set()
        final_docs: list = []

        # Phase 1: top 7 from hop1
        for doc in hop1_docs:
            if len(final_docs) >= 7:
                break
            title = self._doc_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                final_docs.append(doc)

        # Phase 2: top 7 from hop2 (not already seen)
        for doc in hop2_docs:
            if len(final_docs) >= 14:
                break
            title = self._doc_title(doc)
            if title not in seen_titles:
                seen_titles.add(title)
                final_docs.append(doc)

        # Phase 3: round-robin from hop3+hop4 to fill remaining slots (up to 21)
        for i in range(self.k):
            for hop_docs in (hop3_docs, hop4_docs):
                if i < len(hop_docs) and len(final_docs) < 21:
                    title = self._doc_title(hop_docs[i])
                    if title not in seen_titles:
                        seen_titles.add(title)
                        final_docs.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=final_docs[:21])
