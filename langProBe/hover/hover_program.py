import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateClaimQueries(dspy.Signature):
    """A factual claim mentions approximately 3 Wikipedia articles that need to be retrieved.
    Generate exactly 3 distinct search queries, one for each expected Wikipedia article.

    For each query:
    - Explicitly named entity (person, film, show, song, place, etc.): use the name directly
      as a short query (e.g., "Dan Wieden", "Simone Bolelli", "LA Urban Rangers")
    - Described entity (e.g., "the slogan he coined", "the logo alongside it"):
      infer the specific Wikipedia article name (e.g., "Just Do It Nike slogan", "Swoosh Nike logo")
    - Implied entity (e.g., "the detective show she starred in"):
      reason to the Wikipedia article title (e.g., "Mannix TV show")

    CRITICAL: All 3 queries MUST target DIFFERENT Wikipedia articles.
    Keep each query short (1-6 words), similar to a Wikipedia article title."""

    claim: str = dspy.InputField()
    query1: str = dspy.OutputField(desc="search query for 1st Wikipedia article")
    query2: str = dspy.OutputField(desc="search query for 2nd Wikipedia article (different from query1)")
    query3: str = dspy.OutputField(desc="search query for 3rd Wikipedia article (different from query1 and query2)")


class ExtractGapQuery(dspy.Signature):
    """Given a claim and passages already retrieved, find the single most important
    Wikipedia article NOT yet retrieved.

    Check:
    1. CLAIM ENTITIES FIRST: What named entities (person, film, show, place, etc.) are
       mentioned in the claim that do NOT appear in the retrieved passages?
    2. PASSAGE HINTS: What entity do the passages mention that the claim needs?

    Output a short search query (1-6 words) for the most important missing article.
    Do NOT repeat queries that have already been used."""

    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="passages already retrieved from hops 1-3 (or 1-4)")
    already_searched: str = dspy.InputField(desc="queries already used — do NOT repeat these")
    query: str = dspy.OutputField(desc="query for most important missing Wikipedia article")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    """Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program."""

    def __init__(self):
        super().__init__()
        self.k = 25
        self.generate_queries = dspy.ChainOfThought(GenerateClaimQueries)
        self.extract_gap = dspy.ChainOfThought(ExtractGapQuery)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _doc_title(self, doc: str) -> str:
        """Extract normalised title for deduplication (part before ' | ')."""
        return doc.split(" | ")[0].lower().strip()

    def forward(self, claim):
        # STEP 1: Generate 3 targeted queries from the claim in one shot
        cq = self.generate_queries(claim=claim)
        q1, q2, q3 = cq.query1, cq.query2, cq.query3

        # HOPS 1-3: targeted searches for each claim entity
        hop1_docs = self.retrieve_k(q1).passages
        hop2_docs = self.retrieve_k(q2).passages
        hop3_docs = self.retrieve_k(q3).passages

        # Build context for gap analysis
        context_123 = "\n\n".join(hop1_docs[:3] + hop2_docs[:3] + hop3_docs[:3])
        already_searched_123 = f"{q1}; {q2}; {q3}"

        # HOP 4: first gap-fill query
        hop4_query = self.extract_gap(
            claim=claim,
            passages=context_123,
            already_searched=already_searched_123,
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # HOP 5: second gap-fill query
        context_1234 = "\n\n".join(hop1_docs[:2] + hop2_docs[:2] + hop3_docs[:2] + hop4_docs[:2])
        already_searched_1234 = f"{already_searched_123}; {hop4_query}"
        hop5_query = self.extract_gap(
            claim=claim,
            passages=context_1234,
            already_searched=already_searched_1234,
        ).query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # Interleaved round-robin merge with deduplication
        # Prioritize targeted hops (1-3) then gap fills (4-5)
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
