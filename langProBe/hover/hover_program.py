import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class BridgeQuery(dspy.Signature):
    """Generate a Wikipedia search query to find a supporting document that is
    still MISSING for verifying a multi-hop claim.

    The claim connects several entities. The passages already retrieved mention
    other entities that bridge those connections (e.g. a brand advertised in a
    song, a parent company of a product, a place where a road ends). To retrieve
    the missing supporting Wikipedia page, identify the SPECIFIC named entity
    from the retrieved passages that bridges the claim but does not yet have its
    own Wikipedia page in the retrieved set, and build a concise query from that
    entity's name (and a disambiguating word if the name is ambiguous, e.g.
    "Sunkist soft drink", "Microchip Technology company").
    Output only the search query text.
    """

    claim: str = dspy.InputField()
    retrieved_titles: str = dspy.InputField(desc="Titles already retrieved.")
    passages: str = dspy.InputField()
    query: str = dspy.OutputField()


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 12
        self.max_docs = 21

        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Bridge-aware query generators using passage content to find missing entities.
        self.create_query_hop2 = dspy.ChainOfThought(BridgeQuery)
        self.create_query_hop3 = dspy.ChainOfThought(BridgeQuery)
        self.create_query_hop4 = dspy.ChainOfThought(BridgeQuery)

        # Keep a short summary focused on entities/connections that bridge the claim.
        self.summarize = dspy.ChainOfThought(
            "claim,passages->summary",
        )

    def _interleave_dedup(self, hop_doc_lists, max_docs):
        """Round-robin interleave per-hop ranked lists, dedup by title, cap."""
        seen_titles = set()
        merged = []
        max_len = max((len(h) for h in hop_doc_lists), default=0)
        for i in range(max_len):
            for hop_docs in hop_doc_lists:
                if i < len(hop_docs):
                    doc = hop_docs[i]
                    title = doc.split(" | ")[0]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        merged.append(doc)
                        if len(merged) >= max_docs:
                            return merged
        return merged

    def _titles(self, docs):
        return "; ".join(d.split(" | ")[0] for d in docs)

    def forward(self, claim):
        # HOP 1: retrieve directly with the raw claim.
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize(claim=claim, passages=hop1_docs).summary

        # HOP 2: derive a bridge query targeting an entity named in the
        # retrieved passages but not yet retrieved as its own page.
        hop2_query = self.create_query_hop2(
            claim=claim,
            retrieved_titles=self._titles(hop1_docs),
            passages=hop1_docs,
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages if hop2_query else []
        summary_2 = self.summarize(
            claim=claim, passages=hop1_docs + hop2_docs
        ).summary

        # HOP 3: bridge query after considering hop1 + hop2 retrieved titles.
        hop3_query = self.create_query_hop3(
            claim=claim,
            retrieved_titles=self._titles(hop1_docs + hop2_docs),
            passages=hop1_docs + hop2_docs,
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages if hop3_query else []

        # HOP 4: a final bridge query targeting any entity still missing after
        # the first three retrievals (uses all passages gathered so far).
        all_prior_docs = hop1_docs + hop2_docs + hop3_docs
        hop4_query = self.create_query_hop4(
            claim=claim,
            retrieved_titles=self._titles(all_prior_docs),
            passages=all_prior_docs,
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages if hop4_query else []

        merged = self._interleave_dedup(
            [hop1_docs, hop2_docs, hop3_docs, hop4_docs], self.max_docs
        )
        return dspy.Prediction(retrieved_docs=merged)