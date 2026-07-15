import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class SummarizeHop1(dspy.Signature):
    """You are retrieving Wikipedia documents to verify a multi-hop claim.

    Given the claim and the Wikipedia abstract passages retrieved so far, write a
    concise factual summary that:
    1. Extracts the key named entities (people, organizations, places, works,
       events) appearing in the passages and the relationships among them.
    2. Identifies which entities mentioned in the passages (e.g. via phrases
       like "produced by X", "dedicated to Y", "located beside Z", "directed by
       W") still need their OWN Wikipedia article retrieved to verify or refute
       the claim. This is multi-hop retrieval: a passage often names the
       next-hop entity to chase.
    3. Notes any entity that has already been covered vs. one still missing.

    Do not omit named entities that appear in the passages even if they seem
    tangential; the next hop will use them as search targets."""
    claim: str = dspy.InputField()
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    summary: str = dspy.OutputField(desc="Concise entity-focused summary of passages")


class SummarizeHop2(dspy.Signature):
    """You are retrieving Wikipedia documents to verify a multi-hop claim and
    have already completed two retrieval hops.

    Given the claim, the prior summary, the new passages, and the list of
    Wikipedia article titles ALREADY retrieved, write an updated concise factual
    summary that:
    1. Integrates the new passages with the prior summary, tracking which
       supporting entities have now been found and which are still missing.
       IMPORTANT: an entity is "found" ONLY if its exact Wikipedia article title
       appears in retrieved_titles. An entity merely *mentioned* inside a
       passage is NOT yet retrieved.
    2. Emphasize every named entity mentioned in the passages via a relational
       phrase (e.g. "produced by X", "directed by Y", "dedicated to Z",
       "wife of W", "born in P") whose OWN Wikipedia article is NOT in
       retrieved_titles and is needed to verify or refute the claim.
    3. Explicitly list the candidate entities still missing (not in
       retrieved_titles) that should be searched next.

    Be precise about entity names; they will become search queries."""
    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="Summary from the prior hop")
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hop 1 and hop 2")
    summary: str = dspy.OutputField(desc="Updated entity-focused summary explicitly listing missing entities")


class CreateQueryHop2(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Hop 1 retrieved passages using the raw claim (the prior query) and they
    were summarized. Now generate ONE search query for hop 2 that follows the
    multi-hop chain to an entity still missing.

    DO:
    - Pick a named entity mentioned in the summary (often introduced by hop-1
      passages via a relational phrase such as "produced by X",
      "dedicated to Y", "located beside Z", "directed by W", "wife of W")
      whose OWN Wikipedia article is NOT in retrieved_titles.
    - Use the bare entity name; append a single disambiguator (e.g. "film",
      "band", "actor", "company", "place") only when the name is genuinely
      ambiguous.

    DO NOT:
    - Repeat or restate the prior query (the raw claim) or any phrase that
      would just re-retrieve the same articles.
    - Output "none", "no query", "verification complete", an empty string, or
      any meta-commentary. Always output exactly one concrete entity search
      query.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved in hop 1")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued; do not repeat them")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class CreateQueryHop3(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Two hops of retrieval have been completed. Generate ONE search query for
    hop 3 that follows the multi-hop chain to an entity STILL missing.

    DO:
    - From the summaries, identify a named entity mentioned by the retrieved
      passages via a relational phrase (e.g. "produced by X", "dedicated to Y",
      "directed by W", "wife of W", "located beside Z", "born in P") whose OWN
      Wikipedia article is NOT in retrieved_titles and is needed to verify or
      refute the claim.
    - Target that entity's Wikipedia article. Prefer the bare entity name;
      append a single disambiguator only when the name is genuinely ambiguous.
    - If several entities are still missing, pick a DIFFERENT entity than any
      in prior_queries.

    DO NOT:
    - Output "none", "no query", "no missing entity", "verification complete",
      an empty string, or any meta-commentary. Even if the claim looks
      verified, there is almost always a multi-hop neighbour still missing:
      always output exactly one concrete entity search query for that
      neighbour.
    - Restate the whole claim or repeat a prior query; pick a new specific
      missing entity.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    summary_2: str = dspy.InputField(desc="Updated entity-focused summary after hop 2, listing missing entities")
    retrieved_titles: list[str] = dspy.InputField(desc="Wikipedia article titles already retrieved across hop 1 and hop 2")
    prior_queries: list[str] = dspy.InputField(desc="Search queries already issued; do not repeat them")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.final_doc_limit = 21
        self.create_query_hop2 = dspy.ChainOfThought(CreateQueryHop2)
        self.create_query_hop3 = dspy.ChainOfThought(CreateQueryHop3)
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought(SummarizeHop1)
        self.summarize2 = dspy.ChainOfThought(SummarizeHop2)

    @staticmethod
    def _dedup(passages):
        """Deduplicate passages by their leading title ('Title | ...'),
        preserving first-seen order."""
        seen = set()
        unique = []
        for p in passages:
            title = p.split(" | ", 1)[0]
            if title not in seen:
                seen.add(title)
                unique.append(p)
        return unique

    @staticmethod
    def _titles(passages):
        """Extract the leading Wikipedia title from each 'Title | ...' passage."""
        return [p.split(" | ", 1)[0] for p in passages]

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = self._titles(hop1_docs)
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim,
            summary_1=summary_1,
            retrieved_titles=hop1_titles,
            prior_queries=[claim],
        ).query
        if self._is_refusal(hop2_query):
            hop2_docs = []
        else:
            hop2_query = hop2_query.strip()
            hop2_docs = self.retrieve_k(hop2_query).passages
        hop1_2_titles = hop1_titles + self._titles(hop2_docs)
        summary_2 = self.summarize2(
            claim=claim,
            context=summary_1,
            passages=hop2_docs,
            retrieved_titles=hop1_2_titles,
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            retrieved_titles=hop1_2_titles,
            prior_queries=[claim, hop2_query],
        ).query
        if self._is_refusal(hop3_query):
            hop3_docs = []
        else:
            hop3_query = hop3_query.strip()
            hop3_docs = self.retrieve_k(hop3_query).passages

        all_docs = self._dedup(hop1_docs + hop2_docs + hop3_docs)[: self.final_doc_limit]
        return dspy.Prediction(retrieved_docs=all_docs)

    @staticmethod
    def _is_refusal(q):
        """True if the model emitted a degenerate refusal/no-query instead of a
        concrete entity search query. When true, we skip retrieval for that hop
        (rather than issue a junk search that may displace gold docs under the
        21-doc cap). Match is exact (after lowercasing and stripping trailing
        punctuation/whitespace) to avoid false positives on real entity names."""
        if q is None:
            return True
        cleaned = q.strip().lower().rstrip(".!,;:")
        refusal_tokens = {
            "",
            "none",
            "no query",
            "no missing entity",
            "no missing",
            "no further retrieval required",
            "no further query",
            "no further",
            "no additional query",
            "no additional retrieval required",
            "not required",
            "verification complete",
            "claim verified",
            "n/a",
            "na",
            "no entity",
            "no entity needed",
            "no query needed",
            "no query required",
            "completed",
            "done",
        }
        return cleaned in refusal_tokens