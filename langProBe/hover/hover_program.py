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
    2. COVERAGE CHECK — use the "RETRIEVED ARTICLE TITLES" list at the top of retrieved_passages.
       That list contains the EXACT article titles retrieved so far.
       An entity is COVERED only if its exact article title appears in that list.
       WARNING: Do NOT scan the body text (the part after " | ") for coverage — entity names
       frequently appear in other articles' body text without having their own article retrieved.
       Seeing "Jonathan Lynn" mentioned in the Antony Jay article does NOT mean Jonathan Lynn's
       own article is retrieved. Only the title list counts.
       A disambiguation page (title containing "disambiguation") does NOT count as the article.
    3. Output the FIRST named entity from step 1 whose own article title is NOT yet retrieved.
    4. If ALL named entities in the claim already have their own article title retrieved, scan
       the retrieved passage TEXT for implied entities not yet retrieved as their own article
       (e.g., the company that produced a film, the co-winner of an award, the co-author of
       a work, the director of a music video, the composer of a song, the parent company of a
       brand) — output the most important one not yet retrieved.
    5. NEVER search for any query listed in previous_queries — those have already been searched,
       and since retrieval is deterministic, repeating a query CANNOT retrieve new documents.
       If step 3 or step 4 would lead you to repeat a previous query, you MUST instead look for
       a DIFFERENT uncovered entity in the claim or retrieved text.

    Output ONLY a concise Wikipedia article title or entity name — nothing else.
    Good examples: "Pablo Escobar", "Apple Inc.", "Gene Kelly", "Sheldon Lee Glashow",
                   "2004-05 Memphis Grizzlies season", "World Without Love", "Warren Fu"
    Bad examples: "Who starred in Narcos?", "Was Steven Weinberg a professor?"
    Do NOT output a question. Do NOT output a sentence. Output a Wikipedia title or entity name only.
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(
        desc="Wikipedia passages retrieved so far. Starts with 'RETRIEVED ARTICLE TITLES' list, "
             "then full passage text. Coverage check: use ONLY the title list — body text mentions do NOT count."
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

    @staticmethod
    def _get_query_with_retry(identify_fn, claim, retrieved_passages, previous_queries, all_previous_queries):
        """Get the next query from the LM. If it returns an already-searched query, retry once."""
        result = identify_fn(
            claim=claim,
            retrieved_passages=retrieved_passages,
            previous_queries=previous_queries
        )
        query = result.query.strip()
        # If the LM repeated a previous query (which is deterministic and useless), retry once
        if query.lower() in {q.lower() for q in all_previous_queries}:
            prev_str = ", ".join(all_previous_queries)
            retry_result = identify_fn(
                claim=claim,
                retrieved_passages=retrieved_passages,
                previous_queries=f"ALREADY SEARCHED — DO NOT REPEAT ANY OF THESE: {prev_str}. You MUST identify a DIFFERENT uncovered entity."
            )
            query = retry_result.query.strip()
        return query

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

        def build_context(docs):
            if not docs:
                return "No passages retrieved yet."
            # Extract and display titles explicitly so LM can scan them directly
            titles = [doc.split(" | ")[0].strip() for doc in docs]
            title_list = "RETRIEVED ARTICLE TITLES (coverage check: an entity is ONLY covered if its title appears in this list):\n" + "\n".join(f"  - {t}" for t in titles)
            passages_text = "\n---\n".join(docs)
            return f"{title_list}\n\nFULL PASSAGES:\n{passages_text}"

        # HOP 1: Direct retrieval on raw claim
        hop1_new = get_new_unique(self.retrieve_k(claim).passages)

        # HOP 2: Identify the next most important missing entity
        context2 = build_context(hop1_new)
        hop2_query = HoverMultiHop._get_query_with_retry(
            self.identify_hop2_target,
            claim=claim,
            retrieved_passages=context2,
            previous_queries=prev_queries_str(),
            all_previous_queries=all_previous_queries
        )
        hop2_new = get_new_unique(self.retrieve_k(hop2_query).passages, hop2_query)

        # HOP 3: Identify another missing entity (aware of hops 1+2)
        early_docs = hop1_new + hop2_new
        context3 = build_context(early_docs)
        hop3_query = HoverMultiHop._get_query_with_retry(
            self.identify_hop3_target,
            claim=claim,
            retrieved_passages=context3,
            previous_queries=prev_queries_str(),
            all_previous_queries=all_previous_queries
        )
        hop3_new = get_new_unique(self.retrieve_k(hop3_query).passages, hop3_query)

        # HOP 4: Final targeted sweep
        early_docs = hop1_new + hop2_new + hop3_new
        context4 = build_context(early_docs)
        hop4_query = HoverMultiHop._get_query_with_retry(
            self.identify_hop4_target,
            claim=claim,
            retrieved_passages=context4,
            previous_queries=prev_queries_str(),
            all_previous_queries=all_previous_queries
        )
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
