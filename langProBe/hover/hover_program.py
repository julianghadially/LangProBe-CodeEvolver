import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GapFillingEntityName(dspy.Signature):
    """You are given a claim, a list of Wikipedia article titles already retrieved, and a summary
    of what has been found so far. Compare the entities/topics explicitly mentioned in the claim
    against the retrieved titles. Identify the single most important named entity from the claim
    that is NOT yet represented in the retrieved documents.

    Output ONLY the exact entity name as it would appear as a Wikipedia article title.
    Good examples: 'Rogue One', 'University of Florida', 'Caroline Wozniacki', 'Sunkist (soft drink)',
    'Ancient Egyptian religion', 'Home on the Range (2004 film)', 'West Cheshire Association Football League'
    Do NOT output a long descriptive query — output just the entity name."""

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(
        desc="semicolon-separated titles of Wikipedia documents already retrieved in earlier hops"
    )
    summary_2: str = dspy.InputField(
        desc="summary of the context and documents found in hops 1 and 2"
    )
    entity_name: str = dspy.OutputField(
        desc="exact Wikipedia article title of the most important missing entity from the claim"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # k=25 per hop (max allowed). With 3 hops × 25 = 75 candidates, the
        # round-robin naturally goes deeper into each hop when duplicates exist,
        # extending coverage to positions 7-24.
        self.k = 25
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought(GapFillingEntityName)
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

        # HOP 3: gap-filling — identify the exact Wikipedia entity name most likely
        # missing from the claim and search for it directly by title.
        # Using top 8 from each of hops 1+2 for thorough gap analysis.
        retrieved_titles = "; ".join(
            self._doc_title(d) for d in (hop1_docs[:8] + hop2_docs[:8])
        )

        entity_name = self.create_query_hop3(
            claim=claim,
            retrieved_titles=retrieved_titles,
            summary_2=summary_2,
        ).entity_name
        hop3_docs = self.retrieve_k(entity_name).passages

        # Interleaved round-robin merge with deduplication.
        # At each round i, we take position i from hop1, then hop2, then hop3
        # (skipping duplicates by title). With k=25, the loop can go beyond
        # position 7 when inter-hop duplicates arise, extending coverage.
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
