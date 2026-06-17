import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyNextTarget(dspy.Signature):
    """You are selecting the SINGLE next Wikipedia article to retrieve for multi-hop fact verification.

    Given a factual claim and retrieved Wikipedia passages, choose the most important article still missing.

    Steps:
    1. **Claim entities**: List every named entity in the claim (people, places, works, organizations, events).
       - If the claim says "the director/writer/performer/founder of X", include THAT PERSON as an entity, not just X.
       - For seasons, episodes, or specific events, use the EXACT Wikipedia article title (e.g., "2004-05 Memphis Grizzlies season", not "Memphis Grizzlies").

    2. **Coverage check**: For each claim entity, check whether its own dedicated article title appears in retrieved_passages (the text BEFORE " | "). Rules:
       - A mention inside another article's text does NOT count as covered.
       - A disambiguation page does NOT count as the article.

    3. **Second-hop scan** (always run this — not just as a fallback): Read the TEXT of each retrieved passage and identify entities NOT mentioned in the claim that are likely required supporting documents:
       - Co-stars, co-founders, collaborators, or co-authors named in a retrieved person/organization article
       - Films, works, or organizations explicitly named in a retrieved article but not yet retrieved as their own article
       - A broader parent/overview article when you have retrieved specific sub-articles (e.g., if you retrieved articles about specific Egyptian deities or concepts, "Ancient Egyptian religion" may be the required parent article)
       - Named venues, artworks, or institutions the claim describes only obliquely ("the nightclub", "the art installation") — find their proper name in retrieved text and search for it directly

    4. **Select best next query**: From BOTH (a) uncovered claim entities AND (b) second-hop entities from Step 3, choose the SINGLE most important article:
       - ALWAYS prioritize named main subjects (films, games, books, people, organizations) over technical sub-components (engines, concepts, sub-systems). If a game, film, or person AND an engine/concept are both uncovered, retrieve the game/film/person first.
       - If a retrieved passage explicitly names an entity strongly implied by the claim (e.g., a stunt performer's article names the film she worked on; a co-founder's article names their partner), PRIORITIZE that second-hop main entity over remaining claim entities that are less critical to verification
       - Avoid querying specific sub-concepts stated in the claim (e.g., named concepts or sub-events) when a broader parent article is more likely to be a required supporting document
       - Use disambiguation suffixes when needed: "Skittles (confectionery)", "Stranger in Paradise (song)", "Guy Davis (comics)"
       - NEVER repeat a query from previous_queries — ColBERT is deterministic; repeating cannot retrieve new documents. If tempted to repeat, find a DIFFERENT uncovered entity instead.

    Output ONLY a Wikipedia article title or entity name — not a question, not a sentence.
    Good: "Pablo Escobar", "Rogue One", "Ancient Egyptian religion", "2004-05 Memphis Grizzlies season", "Warren Fu"
    Bad: "Who starred in Narcos?", "What is the feather of truth?"
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(
        desc="Wikipedia passages already retrieved (format: 'ArticleTitle | text excerpt...'). "
             "An entity is covered ONLY if its article TITLE (before ' | ') appears here."
    )
    previous_queries: str = dspy.InputField(
        desc="Comma-separated list of queries already searched in prior hops. Do NOT repeat ANY of these — "
             "retrieval is deterministic so repeating a query can NEVER retrieve new documents.",
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
        all_previous_queries = []

        def get_new_unique(docs, query=None):
            """Return only docs with titles not yet seen; track query in all_previous_queries."""
            new = []
            for doc in docs:
                title = doc.split(" | ")[0].strip().lower()
                if title not in seen_titles:
                    seen_titles.add(title)
                    new.append(doc)
            if query is not None:
                all_previous_queries.append(query)
            return new

        def prev_queries_str():
            return ", ".join(all_previous_queries) if all_previous_queries else "None"

        # HOP 1: Direct retrieval on raw claim
        hop1_new = get_new_unique(self.retrieve_k(claim).passages)

        # HOP 2: Identify the next most important missing entity
        context2 = "\n---\n".join(hop1_new) if hop1_new else "No passages retrieved yet."
        hop2_query = self.identify_hop2_target(
            claim=claim,
            retrieved_passages=context2,
            previous_queries=prev_queries_str()
        ).query
        hop2_new = get_new_unique(self.retrieve_k(hop2_query).passages, hop2_query)

        # HOP 3: Identify another missing entity (aware of hops 1+2)
        early_docs = hop1_new + hop2_new
        context3 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop3_query = self.identify_hop3_target(
            claim=claim,
            retrieved_passages=context3,
            previous_queries=prev_queries_str()
        ).query
        hop3_new = get_new_unique(self.retrieve_k(hop3_query).passages, hop3_query)

        # HOP 4: Final targeted sweep
        early_docs = hop1_new + hop2_new + hop3_new
        context4 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop4_query = self.identify_hop4_target(
            claim=claim,
            retrieved_passages=context4,
            previous_queries=prev_queries_str()
        ).query
        hop4_new = get_new_unique(self.retrieve_k(hop4_query).passages, hop4_query)

        # Round-robin interleaving across all 4 hops ensures hop 4 gets proportional
        # slots rather than being starved when hops 1-3 exhaust the 21-doc budget.
        # Takes doc-1 from each hop, then doc-2, etc. No docs are dropped unless
        # all 28 candidates are unique (7 per hop × 4 hops).
        all_hop_docs = [hop1_new, hop2_new, hop3_new, hop4_new]
        interleaved = []
        max_len = max((len(h) for h in all_hop_docs), default=0)
        for i in range(max_len):
            for hop_docs in all_hop_docs:
                if i < len(hop_docs):
                    interleaved.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=interleaved[:21])
