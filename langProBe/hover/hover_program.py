import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class Summarize1Signature(dspy.Signature):
    """Summarize the key entities found in the retrieved passages that are relevant to the claim. Identify which named entities mentioned in the claim appear to be covered by the retrieved passages and which are still missing."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    passages: str = dspy.InputField(desc="Retrieved Wikipedia passages from the first hop")
    summary: str = dspy.OutputField(desc="Summary identifying: (1) which named entities from the claim were found in the passages, and (2) which named entities from the claim are still missing and need to be retrieved next")


class Summarize2Signature(dspy.Signature):
    """Building on the previous context, summarize what additional entities were found in the new passages. Identify which named entities from the claim are still missing after both retrieval rounds."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    context: str = dspy.InputField(desc="Summary from the first retrieval hop identifying what was found and what is missing")
    passages: str = dspy.InputField(desc="New passages retrieved in the second hop")
    summary: str = dspy.OutputField(desc="Summary of: (1) any new entities found in these new passages, and (2) which named entities from the claim are still missing and need to be retrieved")


class QueryHop2Signature(dspy.Signature):
    """Given a factual claim and a summary of initially retrieved passages, generate a short, focused search query to find a specific named entity from the claim that has NOT yet been retrieved. The claim requires finding 3 supporting Wikipedia articles. Focus on a single named entity (person, organization, book, film, song, etc.) that is relevant to the claim but likely missing from the initial retrieval."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(desc="Summary identifying which entities were found and which are still missing after the first hop")
    query: str = dspy.OutputField(desc="A short, focused search query (3-10 words) targeting a specific missing entity from the claim — prefer the entity's proper name as the core of the query")


class QueryHop3Signature(dspy.Signature):
    """Given a factual claim and summaries from two previous retrieval hops, generate a focused search query targeting a specific named entity from the claim that has NOT yet been found. The claim requires 3 supporting Wikipedia articles; identify the one still missing and query for it directly."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(desc="Summary from the first hop identifying what was found and what is missing")
    summary_2: str = dspy.InputField(desc="Summary from the second hop identifying what was found and what is still missing")
    query: str = dspy.OutputField(desc="A short, focused search query (3-10 words) targeting the remaining missing entity — use the entity's proper name as the core of the query")


class QueryHop4GapSignature(dspy.Signature):
    """Given a factual claim and the titles of Wikipedia articles already retrieved, identify which specific named entity from the claim is NOT yet represented in the retrieved documents. Generate a direct, focused query for that entity's Wikipedia article. The claim requires exactly 3 supporting documents — look for the one that is missing."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    retrieved_titles: str = dspy.InputField(desc="Comma-separated list of Wikipedia article titles already retrieved across previous hops")
    query: str = dspy.OutputField(desc="A short, direct query (2-8 words) using the proper name of the specific entity from the claim that is missing from the retrieved titles — e.g., 'Worldview Entertainment film company' or 'Anaïs Nin author'")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15  # More docs per hop for better coverage; max allowed is 25

        # Query generators with entity-focused signatures
        self.create_query_hop2 = dspy.ChainOfThought(QueryHop2Signature)
        self.create_query_hop3 = dspy.ChainOfThought(QueryHop3Signature)
        self.create_query_hop4 = dspy.ChainOfThought(QueryHop4GapSignature)

        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought(Summarize1Signature)
        self.summarize2 = dspy.ChainOfThought(Summarize2Signature)

    def _interleave_and_deduplicate(self, *hop_docs_lists, max_docs=21):
        """Round-robin interleave docs from multiple hops, deduplicate by title, take top max_docs.

        Interleaving ensures each hop contributes roughly equally to the final
        selection, preventing any single hop from dominating the 21-doc budget.
        With 4 hops, each contributes ~5 docs to the final 21.
        """
        seen = set()
        unique = []
        max_len = max((len(lst) for lst in hop_docs_lists), default=0)
        for i in range(max_len):
            for hop_docs in hop_docs_lists:
                if i < len(hop_docs):
                    doc = hop_docs[i]
                    title = doc.split(" | ")[0].strip().lower()
                    if title not in seen:
                        seen.add(title)
                        unique.append(doc)
                        if len(unique) >= max_docs:
                            return unique
        return unique

    def forward(self, claim):
        # HOP 1 - Initial retrieval with raw claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2 - Target a specific entity from the claim not yet fully retrieved
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Target remaining entity not covered by hops 1-2
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # HOP 4 - Gap analysis: look at top retrieved titles, find what's still missing
        # Use interleaved top-20 from hops 1-3 to get a balanced view of what's been retrieved
        combined_so_far = self._interleave_and_deduplicate(
            hop1_docs, hop2_docs, hop3_docs, max_docs=30
        )
        retrieved_titles = ", ".join([d.split(" | ")[0] for d in combined_so_far[:20]])
        hop4_query = self.create_query_hop4(
            claim=claim, retrieved_titles=retrieved_titles
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # Round-robin interleave all hops, deduplicate by title, return top 21
        # Each hop contributes ~5 docs to the final 21 (21 / 4 hops ≈ 5 each)
        return dspy.Prediction(
            retrieved_docs=self._interleave_and_deduplicate(
                hop1_docs, hop2_docs, hop3_docs, hop4_docs, max_docs=21
            )
        )
