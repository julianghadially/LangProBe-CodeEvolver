import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class QueryHop2Signature(dspy.Signature):
    """Given a factual claim and a summary of initially retrieved passages, generate a short, focused search query to find a specific named entity from the claim that has NOT yet been retrieved. The claim requires finding 3 supporting Wikipedia articles. Focus on a single named entity (person, organization, book, film, song, etc.) that is relevant to the claim but likely missing from the initial retrieval."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(desc="Summary of passages retrieved in the first hop, capturing what entities have already been found")
    query: str = dspy.OutputField(desc="A short, focused search query (3-10 words) targeting a specific missing entity from the claim — prefer the entity's proper name as the core of the query")


class QueryHop3Signature(dspy.Signature):
    """Given a factual claim and summaries from two previous retrieval hops, generate a focused search query targeting a specific named entity from the claim that has NOT yet been found. The claim requires 3 supporting Wikipedia articles; identify the one still missing and query for it directly."""

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(desc="Summary of passages from the first hop")
    summary_2: str = dspy.InputField(desc="Summary of passages from the second hop, building on the first")
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
        self.summarize1 = dspy.ChainOfThought("claim, passages -> summary")
        self.summarize2 = dspy.ChainOfThought("claim, context, passages -> summary")

    def _deduplicate(self, docs):
        """Deduplicate docs by normalized title (text before ' | '), preserving order."""
        seen = set()
        unique = []
        for doc in docs:
            title = doc.split(" | ")[0].strip().lower()
            if title not in seen:
                seen.add(title)
                unique.append(doc)
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

        # HOP 4 - Gap analysis: examine already-retrieved titles, find what's still missing
        combined_so_far = self._deduplicate(hop1_docs + hop2_docs + hop3_docs)
        retrieved_titles = ", ".join([d.split(" | ")[0] for d in combined_so_far[:20]])
        hop4_query = self.create_query_hop4(
            claim=claim, retrieved_titles=retrieved_titles
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # Combine all docs, deduplicate by title, and return top 21
        all_docs = self._deduplicate(hop1_docs + hop2_docs + hop3_docs + hop4_docs)
        return dspy.Prediction(retrieved_docs=all_docs[:21])
