import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


def _dedup_by_title(docs, max_docs=21):
    """Deduplicate retrieved passages by normalized Wikipedia title.

    ColBERT returns passages as ``"title | passage text"``; multiple passages
    from the same article share a title, so keeping only the first occurrence
    frees slots for distinct articles. Titles are normalized with
    ``dspy.evaluate.normalize_text`` to match the evaluation metric's title
    equality. Returns at most ``max_docs`` unique passages.
    """
    seen = set()
    unique = []
    for doc in docs:
        title = doc.split(" | ")[0]
        key = dspy.evaluate.normalize_text(title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= max_docs:
            break
    return unique


def _interleave_dedup(hop_docs_list, max_docs=21):
    """Round-robin interleave passages across streams, dedup by normalized title.

    Taking one passage per stream in turn (stream1[0], stream2[0], ...,
    streamN[0], stream1[1], ...) ensures every stream contributes to the final
    ``max_docs``, so diverse discovery streams are not crowded out by earlier
    streams' wider nets. Titles are normalized with
    ``dspy.evaluate.normalize_text`` to match the metric's title equality.
    """
    seen = set()
    unique = []
    if not hop_docs_list:
        return unique
    max_len = max(len(docs) for docs in hop_docs_list)
    for i in range(max_len):
        for docs in hop_docs_list:
            if i >= len(docs):
                continue
            doc = docs[i]
            title = doc.split(" | ")[0]
            key = dspy.evaluate.normalize_text(title)
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
            if len(unique) >= max_docs:
                return unique
    return unique


def _titles(docs):
    """Return the raw Wikipedia title (text before ``" | "``) of each passage."""
    return [doc.split(" | ")[0] for doc in docs]


class DecomposeClaim(dspy.Signature):
    """Decompose the claim into three distinct Wikipedia search queries, each targeting a different key entity or referent.

    HoVer claims chain multiple entities through facts. Identify the distinct named entities -- people, places, organizations, works (films, books, albums), events -- and referents such as "the director", "this religion", "the attraction" that point to one specific Wikipedia article. For each, form a concise search query using the entity's exact name or the most specific phrasing that would retrieve its article.

    The three queries MUST target DIFFERENT entities. Order them by centrality to the claim (most central first). Strip filler words and factual assertions; the goal is a clean retrieval query, not a restatement of the claim."""

    claim = dspy.InputField()
    query1 = dspy.OutputField(desc="A concise search query for the most central entity or referent in the claim.")
    query2 = dspy.OutputField(desc="A concise search query targeting a DIFFERENT entity or referent.")
    query3 = dspy.OutputField(desc="A concise search query targeting a THIRD distinct entity or referent.")


class AdaptiveDiscover(dspy.Signature):
    """Generate ONE search query for a Wikipedia supporting article that is still MISSING from the retrieved titles.

    Some missing articles are entities DISCOVERED in the retrieved passages but not named in the claim -- e.g. a film's director or co-star, an author's collaborator, a company's product, or a place linked to a covered entity. Others are entities NAMED or IMPLIED in the claim whose article the earlier queries failed to retrieve -- e.g. the claim's main subject, a named concept, or a referent such as "this religion" or "the attraction" that points to one specific Wikipedia article. BOTH kinds matter: any article still missing from retrieved_titles may be the gold document, regardless of whether its entity appears in the claim.

    The passages may include results from a PRIOR discovery step. This lets you reach entities that are TWO hops from the claim -- an entity named only in a supporting article's passage, not in the claim or the first-round results. Scan ALL accumulated passages for named entities that could support an as-yet-unsupported part of the claim, including entities you only know about because a prior discovery step retrieved their article.

    Steps:
    1. Read the claim and all retrieved passages (including any prior discovery passages). List candidate Wikipedia articles (people, works, organizations, places, concepts) that could support the claim.
    2. Remove only candidates already present in retrieved_titles (already retrieved). Do NOT discard a candidate merely because it is named in the claim -- if its article is still missing, it is exactly what may be needed.
    3. From the remaining missing candidates -- claim-named, claim-implied, or passage-discovered (including 2-hop entities visible only in prior discovery passages) -- pick the ONE most likely to be the specific Wikipedia article that supports an as-yet-unsupported part of the claim.
    4. Form a concise query using the entity's exact name as a Wikipedia article title would. Query the SPECIFIC article (a particular film, ride, place, person, or concept), not a broad category, franchise, or multi-word descriptive phrase; never re-query a title already in retrieved_titles."""

    claim = dspy.InputField()
    passages = dspy.InputField(desc="All Wikipedia passages retrieved so far (decomposition + any prior discovery) as 'title | passage text'; read their content to discover entities, including 2-hop entities only visible in prior discovery passages.")
    retrieved_titles = dspy.InputField(desc="Wikipedia article titles already retrieved; the query must target an article NOT in this list.")
    query = dspy.OutputField(desc="A concise search query (a Wikipedia article title) targeting one still-missing supporting article.")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        # Phase 1: one-shot parallel claim decomposition (replaces the
        # sequential hop1->summarize->hop2->summarize->hop3 chain, cutting the
        # LM-call chain from 5 to 2 calls to reduce run-to-run variance).
        self.decompose = dspy.ChainOfThought(DecomposeClaim)
        # Phase 2: adaptive discovery chaining -- two sequential steps, each
        # ONE LM call reading all accumulated passages (incl. prior discovery
        # passages) and emitting ONE query.  Step 2 reads step 1's discovery
        # passages, reaching 2-hop entities the single-pass parallel discovery
        # cannot (validated in iter 9: ex 30 Additi Gupta -> Ishqbaaaz fixed).
        self.discover_step = dspy.ChainOfThought(AdaptiveDiscover)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # PHASE 1 -- parallel claim decomposition: enumerate ALL key entities in
        # a single LM call, then retrieve each independently.  No sequential
        # summarization step (which loses information and adds variance); the
        # decomposition reads the claim directly and targets every entity at
        # once.
        decomp = self.decompose(claim=claim)
        decomp_queries = [decomp.query1, decomp.query2, decomp.query3]
        decomp_docs = [self.retrieve_k(q).passages for q in decomp_queries]

        # PHASE 2 -- adaptive discovery chaining: two sequential steps, each
        # reads ALL accumulated passages (incl. prior discovery passages) and
        # emits ONE query for a still-missing supporting article.  Step 2
        # reads step 1's discovery passages, reaching 2-hop entities (e.g. an
        # entity named only in a supporting article's passage, not in the claim
        # or the decomposition results) that the single-pass parallel discovery
        # structurally cannot reach.
        prior_unique = _dedup_by_title(
            decomp_docs[0] + decomp_docs[1] + decomp_docs[2], max_docs=21
        )
        disc_a = self.discover_step(
            claim=claim,
            passages=prior_unique,
            retrieved_titles=_titles(prior_unique),
        )
        disc_a_docs = self.retrieve_k(disc_a.query).passages

        # Step 2 reads step 1's discovery passages too; cap 31 (not 21) so the
        # new discovery titles reach step 2's view (a 21 cap would let
        # prior_unique fill the slots and crowd out disc_a_docs).
        step2_passages = _dedup_by_title(
            prior_unique + disc_a_docs, max_docs=31
        )
        disc_b = self.discover_step(
            claim=claim,
            passages=step2_passages,
            retrieved_titles=_titles(step2_passages),
        )
        disc_b_docs = self.retrieve_k(disc_b.query).passages

        # Round-robin interleave so all five streams contribute to the final 21
        # slots; hop-order dedup would let the decomposition streams fill the
        # cap and crowd out the two discovery streams whose gold typically ranks
        # top-1-3.
        retrieved_docs = _interleave_dedup(
            decomp_docs + [disc_a_docs, disc_b_docs], max_docs=21
        )
        return dspy.Prediction(retrieved_docs=retrieved_docs)
