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


def _weighted_interleave_dedup(hop_docs_list, max_per_stream, max_docs=21):
    """Round-robin interleave with per-stream slot caps, dedup by normalized title.

    Like ``_interleave_dedup`` but each stream ``i`` is capped at
    ``max_per_stream[i]`` unique passages. This lets proven streams (e.g.
    decomposition, whose golds may rank beyond position 2) keep more slots when
    new discovery streams are added, while the new streams (whose golds
    typically rank top-1) get fewer slots. Titles are normalized with
    ``dspy.evaluate.normalize_text`` to match the metric's title equality.
    """
    seen = set()
    unique = []
    if not hop_docs_list:
        return unique
    taken = [0] * len(hop_docs_list)
    max_len = max(len(docs) for docs in hop_docs_list)
    for i in range(max_len):
        for s, docs in enumerate(hop_docs_list):
            if i >= len(docs):
                continue
            if taken[s] >= max_per_stream[s]:
                continue
            doc = docs[i]
            title = doc.split(" | ")[0]
            key = dspy.evaluate.normalize_text(title)
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
            taken[s] += 1
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


class DiscoverMissingEntities(dspy.Signature):
    """Generate TWO distinct search queries for Wikipedia supporting articles that are still MISSING from the retrieved titles.

    Some missing articles are entities DISCOVERED in the retrieved passages but not named in the claim -- e.g. a film's director or co-star, an author's collaborator, a company's product, or a place linked to a covered entity. Others are entities NAMED or IMPLIED in the claim whose article the earlier hops failed to retrieve -- e.g. the claim's main subject, a named concept, or a referent such as "this religion" or "the attraction" that points to one specific Wikipedia article. BOTH kinds matter: any article still missing from retrieved_titles may be the gold document, regardless of whether its entity appears in the claim.

    A single guess is often wrong, so issue queries for TWO different missing articles to raise the chance of recovering the gold document.

    Steps:
    1. Read the claim and the retrieved passages. List candidate Wikipedia articles (people, works, organizations, places, concepts) that could support the claim.
    2. Remove only candidates already present in retrieved_titles (already retrieved). Do NOT discard a candidate merely because it is named in the claim -- if its article is still missing, it is exactly what may be needed.
    3. From the remaining missing candidates -- claim-named, claim-implied, or passage-discovered -- pick the TWO most likely to be the specific Wikipedia article that supports an as-yet-unsupported part of the claim. The two queries MUST target DIFFERENT entities (not two phrasings of the same entity).
    4. Form a concise query for each using the entity's exact name as a Wikipedia article title would. Query the SPECIFIC article (a particular film, ride, place, person, or concept), not a broad category, franchise, or multi-word descriptive phrase; never re-query a title already in retrieved_titles."""

    claim = dspy.InputField()
    passages = dspy.InputField(desc="Wikipedia passages retrieved in the decomposition phase as 'title | passage text'; read their content to discover entities.")
    retrieved_titles = dspy.InputField(desc="Wikipedia article titles already retrieved; both queries must target articles NOT in this list.")
    query1 = dspy.OutputField(desc="A concise search query (a Wikipedia article title) targeting one still-missing supporting article.")
    query2 = dspy.OutputField(desc="A concise search query (a Wikipedia article title) targeting a DIFFERENT still-missing supporting article.")


class AdaptiveDiscover(dspy.Signature):
    """Generate TWO distinct search queries for Wikipedia supporting articles that are still MISSING from the retrieved titles.

    Some missing articles are entities DISCOVERED in the retrieved passages but not named in the claim -- e.g. a film's director or co-star, an author's collaborator, a company's product, or a place linked to a covered entity. Others are entities NAMED or IMPLIED in the claim whose article the earlier queries failed to retrieve -- e.g. the claim's main subject, a named concept, or a referent such as "this religion" or "the attraction" that points to one specific Wikipedia article. BOTH kinds matter: any article still missing from retrieved_titles may be the gold document, regardless of whether its entity appears in the claim.

    The passages may include results from a PRIOR discovery step. This lets you reach entities that are TWO hops from the claim -- an entity named only in a supporting article's passage, not in the claim or the first-round results. Scan ALL accumulated passages for named entities that could support an as-yet-unsupported part of the claim, including entities you only know about because a prior discovery step retrieved their article.

    Steps:
    1. Read the claim and all retrieved passages (including any prior discovery passages). SCAN the passage text for PROPER NOUNS -- specific person names, place names, organization names, and work titles (films, books, albums, events) -- that appear in the passages. Each proper noun likely has its own Wikipedia article that could support the claim; these are the strongest candidates. Also consider claim-named or claim-implied entities whose articles are still missing.
    2. Remove only candidates already present in retrieved_titles (already retrieved). Do NOT discard a candidate merely because it is named in the claim -- if its article is still missing, it is exactly what may be needed.
    3. From the remaining missing candidates -- prioritizing proper-noun entities found in the passage text (including 2-hop entities visible only in prior discovery passages) -- pick the TWO most likely to be the specific Wikipedia articles that support as-yet-unsupported parts of the claim. The two queries MUST target DIFFERENT entities (not two phrasings of the same entity). Do NOT query for a role, relationship, or conceptual description -- instead, identify the SPECIFIC person or entity that fills that role by reading the passage, and query for that entity's exact name.
    4. Form a concise query for each using the entity's exact name as a Wikipedia article title would. Query the SPECIFIC article (a particular film, ride, place, person, or concept), not a broad category, franchise, or multi-word descriptive phrase; never re-query a title already in retrieved_titles."""

    claim = dspy.InputField()
    passages = dspy.InputField(desc="All Wikipedia passages retrieved so far (decomposition + any prior discovery) as 'title | passage text'; read their content to discover entities, including 2-hop entities only visible in prior discovery passages.")
    retrieved_titles = dspy.InputField(desc="Wikipedia article titles already retrieved; both queries must target articles NOT in this list.")
    query1 = dspy.OutputField(desc="A concise search query (a Wikipedia article title) targeting one still-missing supporting article.")
    query2 = dspy.OutputField(desc="A concise search query (a Wikipedia article title) targeting a DIFFERENT still-missing supporting article.")


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
        # Phase 2: hybrid discovery.  Step 1 keeps iter 6's proven 2-query
        # single-pass discovery (the 2-query hedge recovers 1-level golds even
        # when one query mis-targets via high-salience bias).  Step 2 is a
        # chaining step that reads ALL accumulated passages (incl. step 1's
        # discovery results) and emits TWO queries, reaching 2-hop entities the
        # single-pass discovery structurally cannot (validated iter 9: ex 30
        # Additi Gupta -> Ishqbaaaz fixed; iter 10: ex 97 Gene Kelly found).
        # The 2-query hedge mirrors step 1: a single chaining query has no
        # recovery from a mis-target (iter 10 round 1 lesson).
        self.discover = dspy.ChainOfThought(DiscoverMissingEntities)
        self.discover_chain = dspy.ChainOfThought(AdaptiveDiscover)
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

        # PHASE 2 -- hybrid discovery.  Step 1: iter 6's proven 2-query
        # single-pass discovery reads the decomposition passages and emits TWO
        # queries for still-missing supporting articles (the 2-query hedge
        # recovers 1-level golds even when one query mis-targets).  Step 2: a
        # chaining step reads ALL accumulated passages (incl. step 1's
        # discovery results) and emits TWO queries, reaching 2-hop entities (an
        # entity named only in a supporting article's passage, not in the claim
        # or the decomposition results) the single-pass discovery cannot.  The
        # 2-query hedge mirrors step 1's proven design: a single chaining query
        # has no recovery from a mis-target.
        prior_unique = _dedup_by_title(
            decomp_docs[0] + decomp_docs[1] + decomp_docs[2], max_docs=21
        )
        disc = self.discover(
            claim=claim,
            passages=prior_unique,
            retrieved_titles=_titles(prior_unique),
        )
        disc_a_docs = self.retrieve_k(disc.query1).passages
        disc_b_docs = self.retrieve_k(disc.query2).passages

        # Chaining step: cap 31 (not 21) so step 1's new discovery titles reach
        # the chaining step's view (a 21 cap would let prior_unique fill the
        # slots and crowd out the discovery results).
        chain_passages = _dedup_by_title(
            prior_unique + disc_a_docs + disc_b_docs, max_docs=31
        )
        disc_c = self.discover_chain(
            claim=claim,
            passages=chain_passages,
            retrieved_titles=_titles(chain_passages),
        )
        disc_c_docs = self.retrieve_k(disc_c.query1).passages
        disc_d_docs = self.retrieve_k(disc_c.query2).passages

        # Weighted round-robin interleave: decomposition streams get 4 slots
        # (preserving iter 6/10's allocation — decomp golds may rank at
        # position 3), discovery streams get 3 (same as iter 6's 5-stream
        # design), and chaining streams get 2 (their golds are targeted queries
        # that typically rank top-1).  This prevents the 7th stream from
        # reducing decomp slots (4->3 in an equal 7-way round-robin) and the
        # associated slot-loss risk.  Hop-order dedup would let the
        # decomposition streams fill the cap and crowd out the discovery
        # streams.
        retrieved_docs = _weighted_interleave_dedup(
            decomp_docs + [disc_a_docs, disc_b_docs, disc_c_docs, disc_d_docs],
            max_per_stream=[4, 4, 4, 3, 3, 2, 2],
            max_docs=21,
        )
        return dspy.Prediction(retrieved_docs=retrieved_docs)
