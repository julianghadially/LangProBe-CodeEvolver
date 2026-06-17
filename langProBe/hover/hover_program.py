import dspy
import unicodedata
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
       - Include ALL named entities, even those used only as comparison subjects or secondary
         referents (e.g., "more scope than Robert E. Howard" → Robert E. Howard is required;
         "born before X" → X is required; "partner of Y" → Y is required).
    2. For EACH named entity from step 1, check the retrieved_passages for a passage whose
       ARTICLE TITLE (the text before the " | " separator) matches that entity's name.
       IMPORTANT: An entity is covered ONLY if its own dedicated article title appears —
       a mere mention of the entity inside another article's text does NOT count as covered.
       Concrete example: if you retrieve "Person A | ...text mentioning Person B...", that does
       NOT cover Person B. You still need a passage starting with "Person B | ..." to cover
       Person B. Always check the TITLE before " | ", never the body text, for coverage.
       A disambiguation page (title containing "disambiguation") does NOT count as the article.
    3. Output the FIRST named entity from step 1 whose own article title is NOT yet retrieved.
    4. If ALL named entities in the claim already have their own article title retrieved, scan
       the retrieved passage TEXT for implied entities not yet retrieved as their own article
       (e.g., the company that produced a film, the co-winner of an award, the co-author of
       a work, the director of a music video, the composer of a song, the parent company of a
       brand, the 1975 film that inspired a TV show). When you identify such an implied entity
       in the retrieved text, OUTPUT THAT IMPLIED ENTITY AS YOUR QUERY IMMEDIATELY — identifying
       it in the text does NOT mean it is covered, and you MUST query it as a new search before
       moving to any other entity.
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
        self.identify_hop5_target = dspy.ChainOfThought(IdentifyNextTarget)

    @staticmethod
    def _normalize_query(q: str) -> str:
        """Normalize a query string for duplicate detection."""
        q = q.strip().lower()
        # Normalize em-dashes, en-dashes to regular hyphens for fuzzy comparison
        q = q.replace('–', '-').replace('—', '-')
        q = unicodedata.normalize('NFC', q)
        return q

    @staticmethod
    def _get_query_with_retry(predictor, claim, retrieved_passages, previous_queries_str):
        """Call predictor; if query is a duplicate of a prior query, retry once with an explicit warning."""
        result = predictor(
            claim=claim,
            retrieved_passages=retrieved_passages,
            previous_queries=previous_queries_str,
        )
        query = result.query.strip()

        # Check for duplicate using normalized comparison
        prev_set = {
            HoverMultiHop._normalize_query(q)
            for q in previous_queries_str.split(",")
            if q.strip() and q.strip() not in ("None", "")
        }
        if HoverMultiHop._normalize_query(query) in prev_set:
            # Retry once with an explicit CRITICAL warning injected into previous_queries
            augmented_prev = (
                previous_queries_str
                + f", [CRITICAL: '{query}' was already searched — you MUST choose a DIFFERENT uncovered entity]"
            )
            result = predictor(
                claim=claim,
                retrieved_passages=retrieved_passages,
                previous_queries=augmented_prev,
            )
            query = result.query.strip()
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

        # HOP 1: Direct retrieval on raw claim
        hop1_new = get_new_unique(self.retrieve_k(claim).passages)

        # HOP 2: Identify the next most important missing entity
        context2 = "\n---\n".join(hop1_new) if hop1_new else "No passages retrieved yet."
        hop2_query = self._get_query_with_retry(
            self.identify_hop2_target, claim, context2, prev_queries_str()
        )
        hop2_new = get_new_unique(self.retrieve_k(hop2_query).passages, hop2_query)

        # HOP 3: Identify another missing entity (aware of hops 1+2)
        early_docs = hop1_new + hop2_new
        context3 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop3_query = self._get_query_with_retry(
            self.identify_hop3_target, claim, context3, prev_queries_str()
        )
        hop3_new = get_new_unique(self.retrieve_k(hop3_query).passages, hop3_query)

        # HOP 4: Deep chain completion
        early_docs = hop1_new + hop2_new + hop3_new
        context4 = "\n---\n".join(early_docs) if early_docs else "No passages retrieved yet."
        hop4_query = self._get_query_with_retry(
            self.identify_hop4_target, claim, context4, prev_queries_str()
        )
        hop4_new = get_new_unique(self.retrieve_k(hop4_query).passages, hop4_query)

        # HOP 5: Final bridging hop — catches entities that require one more step after hop 4
        all_docs = hop1_new + hop2_new + hop3_new + hop4_new
        context5 = "\n---\n".join(all_docs) if all_docs else "No passages retrieved yet."
        hop5_query = self._get_query_with_retry(
            self.identify_hop5_target, claim, context5, prev_queries_str()
        )
        hop5_new = get_new_unique(self.retrieve_k(hop5_query).passages, hop5_query)

        # Round-robin interleaving across all 5 hops ensures each hop gets proportional
        # slots rather than being starved when other hops exhaust the 21-doc budget.
        # Takes doc-1 from each hop, then doc-2, etc. No docs are dropped unless
        # all 35 candidates are unique (7 per hop × 5 hops).
        all_hop_docs = [hop1_new, hop2_new, hop3_new, hop4_new, hop5_new]
        interleaved = []
        max_len = max((len(h) for h in all_hop_docs), default=0)
        for i in range(max_len):
            for hop_docs in all_hop_docs:
                if i < len(hop_docs):
                    interleaved.append(hop_docs[i])

        return dspy.Prediction(retrieved_docs=interleaved[:21])
