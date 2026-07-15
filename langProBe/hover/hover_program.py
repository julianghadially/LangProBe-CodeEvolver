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

    Given the claim, the prior summary, and the new passages, write an updated
    concise factual summary that:
    1. Integrates the new passages with the prior summary, tracking which
       supporting entities have now been found and which are still missing.
    2. Emphasizes any named entity mentioned in the passages whose OWN
       Wikipedia article has not yet been retrieved and is needed to verify or
       refute the claim (multi-hop chain following).
    3. Lists the entity that should be searched next and why.

    Be precise about entity names; they will become search queries."""
    claim: str = dspy.InputField()
    context: str = dspy.InputField(desc="Summary from the prior hop")
    passages: list[str] = dspy.InputField(desc="Wikipedia abstract passages retrieved this hop")
    summary: str = dspy.OutputField(desc="Updated entity-focused summary")


class CreateQueryHop2(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Hop 1 retrieved passages using the raw claim and they were summarized. Now
    generate ONE search query for hop 2 that follows the multi-hop chain:
    - Pick a named entity that appears in the summary (often introduced by the
      hop-1 passages via a relational phrase such as "produced by X",
      "dedicated to Y", "located beside Z") but whose own Wikipedia article
      has not yet been retrieved.
    - The query should target that entity's Wikipedia article. Prefer the bare
      entity name; append a single disambiguator (e.g. "film", "band",
      "actor", "company", "place") only when the name is genuinely ambiguous.
    - Do NOT restate the whole claim or search for the claim text itself; pick
      the specific missing entity.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    query: str = dspy.OutputField(desc="A single targeted Wikipedia search query for one missing entity")


class CreateQueryHop3(dspy.Signature):
    """You are guiding multi-hop Wikipedia retrieval to verify a claim.

    Two hops of retrieval have been completed. Generate ONE search query for
    hop 3 that follows the multi-hop chain to an entity still missing:
    - From the summaries, identify a named entity mentioned by the retrieved
      passages (e.g. via "produced by X", "dedicated to Y", "directed by W",
      "located beside Z") whose own Wikipedia article has not yet been
      retrieved and is needed to verify or refute the claim.
    - Target that entity's Wikipedia article. Prefer the bare entity name;
      append a single disambiguator only when the name is genuinely ambiguous.
    - Pick a DIFFERENT entity than hop 2 if multiple are missing.
    - Do NOT restate the whole claim; pick the specific missing entity.

    Output a single concise search query."""
    claim: str = dspy.InputField()
    summary_1: str = dspy.InputField(desc="Entity-focused summary of hop-1 passages")
    summary_2: str = dspy.InputField(desc="Updated entity-focused summary after hop 2")
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

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        all_docs = self._dedup(hop1_docs + hop2_docs + hop3_docs)[: self.final_doc_limit]
        return dspy.Prediction(retrieved_docs=all_docs)