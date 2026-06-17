import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyNextTarget(dspy.Signature):
    """You are retrieving Wikipedia articles to support verification of a multi-hop factual claim.

    Given the claim and Wikipedia passages already retrieved, identify the SINGLE most important
    Wikipedia article still needed to verify the claim.

    Steps:
    1. List ALL named entities explicitly mentioned in the claim (people, places, organizations,
       works, songs, films, awards, titles, etc.). CRITICAL nuances:
       - If the claim references "the person who wrote/directed/performed/created X", the PERSON
         themselves is a required entity (not just X's article). E.g., if the claim says
         "fronted by [person]", that person's own article is needed; if the claim says
         "the director of [film]", that director's own article is needed.
       - If the claim references a specific season, episode, or event, use the EXACT Wikipedia
         article title (e.g., "2004-05 Memphis Grizzlies season" not just "Memphis Grizzlies";
         "World Without Love" as a song article, not just "Peter and Gordon").
    2. For EACH named entity from step 1, check the retrieved_passages for a passage whose
       ARTICLE TITLE (the text before the " | " separator) matches that entity's name.
       IMPORTANT: An entity is covered ONLY if its own dedicated article title appears —
       a mere mention of the entity inside another article's text does NOT count as covered.
       A disambiguation page (title containing "disambiguation") does NOT count as the article.
    3. Output the FIRST named entity from step 1 whose own article title is NOT yet retrieved.
    4. If ALL named entities in the claim already have their own article title retrieved, scan
       the retrieved passage TEXT for implied entities not yet retrieved as their own article
       (e.g., the company that produced a film, the co-winner of an award, the co-author of
       a work, the director of a music video, the composer of a song, the parent company of a
       brand) — output the most important one not yet retrieved.
    5. NEVER repeat a query listed in fruitless_queries (those returned 0 new documents).
       If step 3 or step 4 would lead you to repeat a fruitless query, instead look for a
       DIFFERENT uncovered entity in the claim or retrieved text.

    Output ONLY a concise Wikipedia article title or entity name — nothing else.
    Good examples: "Pablo Escobar", "Apple Inc.", "Gene Kelly", "Sheldon Lee Glashow",
                   "2004-05 Memphis Grizzlies season", "World Without Love", "Warren Fu"
    Bad examples: "Who starred in Narcos?", "Was Steven Weinberg a professor?"
    Do NOT output a question. Do NOT output a sentence. Output a Wikipedia title or entity name only.
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(
        desc="Wikipedia passages already retrieved (format: 'ArticleTitle | text excerpt...'). "
             "An entity is covered ONLY if its article TITLE (before ' | ') appears here."
    )
    fruitless_queries: str = dspy.InputField(
        desc="Comma-separated queries that were searched but returned 0 new unique documents. Do NOT repeat any of these exact queries.",
        default="None"
    )
    query: str = dspy.OutputField(
        desc="A single Wikipedia article title or entity name to search for next — "
             "NOT a question, NOT a sentence"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.identify_hop2_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop3_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop4_target = dspy.ChainOfThought(IdentifyNextTarget)

    def forward(self, claim):
        seen_titles = set()
        fruitless_queries = []

        def get_new_unique(docs, query=None):
            """Return only docs with titles not yet seen; flag query as fruitless if 0 new docs."""
            new = []
            for doc in docs:
                title = doc.split(" | ")[0].strip().lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    new.append(doc)
            if query is not None and not new:
                fruitless_queries.append(query)
            return new

        def fruitless_str():
            return ", ".join(fruitless_queries) if fruitless_queries else "None"

        # HOP 1: Direct retrieval on raw claim
        hop1_new = get_new_unique(self.retrieve_k(claim).passages)

        # HOP 2: Identify the next most important missing entity
        context2 = "\n---\n".join(hop1_new) if hop1_new else "No passages retrieved yet."
        hop2_query = self.identify_hop2_target(
            claim=claim,
            retrieved_passages=context2,
            fruitless_queries=fruitless_str()
        ).query
        hop2_new = get_new_unique(self.retrieve_k(hop2_query).passages, hop2_query)

        # HOP 3: Identify another missing entity (aware of hops 1+2)
        early_docs = hop1_new + hop2_new
        context3 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop3_query = self.identify_hop3_target(
            claim=claim,
            retrieved_passages=context3,
            fruitless_queries=fruitless_str()
        ).query
        hop3_new = get_new_unique(self.retrieve_k(hop3_query).passages, hop3_query)

        # HOP 4: Final targeted sweep — guaranteed slots via slot reservation
        early_docs = hop1_new + hop2_new + hop3_new
        context4 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop4_query = self.identify_hop4_target(
            claim=claim,
            retrieved_passages=context4,
            fruitless_queries=fruitless_str()
        ).query
        hop4_new = get_new_unique(self.retrieve_k(hop4_query).passages, hop4_query)

        # Slot allocation: hops 1-3 get at most 14 slots, hop 4 gets the remainder (up to 7).
        # This guarantees hop 4's new unique documents are included even when early hops
        # fill 21 total unique docs (the previous "slot starvation" bug).
        final_docs = early_docs[:14] + hop4_new

        return dspy.Prediction(retrieved_docs=final_docs[:21])
