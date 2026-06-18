import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class Summarize1Signature(dspy.Signature):
    """Analyze the retrieved passages and track entity coverage for the claim.

    CRITICAL DISTINCTION — an entity is only 'DIRECTLY RETRIEVED' if its OWN
    Wikipedia article appears in the passages (the document title begins with that
    entity's name). Finding an entity MENTIONED inside another article does NOT
    count as retrieving that entity's Wikipedia page — it still needs to be
    directly searched by name.

    Example: retrieving 'The Four-Chambered Heart | novel by Anaïs Nin...' means
    you have the book's article. You have NOT retrieved the 'Anaïs Nin' article —
    her biography page still needs to be fetched.

    Example: retrieving 'Capitale de la douleur | collection by Paul Éluard...'
    means you have the book's article but NOT 'Paul Éluard' — his biography page
    must still be searched directly.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    passages: str = dspy.InputField(
        desc="Retrieved Wikipedia passages from the first hop (format: 'Title | text')"
    )
    summary: str = dspy.OutputField(
        desc=(
            "Coverage report listing: "
            "(1) DIRECTLY RETRIEVED: entities whose OWN Wikipedia article title appears in passages; "
            "(2) MENTIONED BUT NOT RETRIEVED: entities named inside retrieved articles but lacking their own fetched page; "
            "(3) NOT YET FOUND: entities from the claim absent entirely. "
            "Be specific about entity names for category (2) — these are the top priority for next queries."
        )
    )


class Summarize2Signature(dspy.Signature):
    """Update entity coverage after the second retrieval hop.

    CRITICAL DISTINCTION — an entity is only 'DIRECTLY RETRIEVED' if its OWN
    Wikipedia article title appears in the passages (title begins with that entity's
    name). Being merely mentioned inside another article does NOT count — that
    entity's own page still needs a direct search.

    Check both the new passages AND re-examine prior context for any entities that
    were mentioned in earlier passes but whose own article was never fetched.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    context: str = dspy.InputField(
        desc="Coverage report from the first hop listing directly-retrieved, mentioned-but-not-retrieved, and missing entities"
    )
    passages: str = dspy.InputField(
        desc="New passages retrieved in the second hop (format: 'Title | text')"
    )
    summary: str = dspy.OutputField(
        desc=(
            "Updated coverage report: "
            "(1) DIRECTLY RETRIEVED: entities with their own Wikipedia article now in hand; "
            "(2) MENTIONED BUT NOT RETRIEVED: entities named in retrieved articles but whose own page is still missing — list their EXACT names; "
            "(3) NOT YET FOUND: entities from the claim not seen at all. "
            "The next query should directly target an entity from category (2) by its proper name."
        )
    )


class QueryHop2Signature(dspy.Signature):
    """Generate a focused search query to directly retrieve a missing entity's Wikipedia article.

    PRIORITY RULE: If any entity was MENTIONED inside a retrieved passage but its
    own Wikipedia article was not retrieved, query DIRECTLY for that entity by its
    proper name. Use the entity's name as the core of the query — this is far more
    reliable than descriptive queries like 'author of book X'.

    BAD:  'author of Capitale de la douleur' (descriptive, ColBERT may miss the article)
    GOOD: 'Paul Éluard French poet' (direct name query, will surface the biography)

    BAD:  'director of Leslie Nielsen comedy' (descriptive)
    GOOD: 'Allan Goldstein film director' (direct name query)

    IMPORTANT: The claim may contain misleading or incorrect details (wrong nationality,
    wrong location, etc.). Focus on what entities' Wikipedia articles are structurally
    needed, not on verifying the claim's specific wording.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(
        desc="Coverage report from hop 1 — check 'MENTIONED BUT NOT RETRIEVED' category for priority targets"
    )
    previous_queries: str = dspy.InputField(
        desc="Queries already used in previous hops — do NOT generate a query identical or very similar to any of these"
    )
    query: str = dspy.OutputField(
        desc=(
            "A short, focused search query (2-8 words). "
            "If an entity is mentioned-but-not-retrieved, use its PROPER NAME directly. "
            "Do not repeat previous_queries."
        )
    )


class QueryHop3Signature(dspy.Signature):
    """Generate a focused search query to retrieve the next missing entity's Wikipedia article.

    PRIORITY RULE: Check the summaries for entities listed under 'MENTIONED BUT NOT
    RETRIEVED' — these are entities named in retrieved passages whose own Wikipedia
    article has never been fetched. Query directly for them by proper name.

    IMPORTANT: The claim may contain typos or misleading details. If the claim has
    an unusual name that didn't produce results, try alternative spellings or use
    other context clues from the retrieved passages to identify the correct entity.
    For example, 'Charpes Lane' is likely 'Charles Lane'; use retrieved context
    (e.g., film titles, co-stars) to deduce the correct name.

    Do NOT generate a query that duplicates or closely resembles a previous query.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    summary_1: str = dspy.InputField(desc="Coverage report from the first hop")
    summary_2: str = dspy.InputField(
        desc="Updated coverage report from the second hop — check 'MENTIONED BUT NOT RETRIEVED' for priority targets"
    )
    previous_queries: str = dspy.InputField(
        desc="Semicolon-separated list of queries already used — do NOT repeat these"
    )
    query: str = dspy.OutputField(
        desc=(
            "A short, focused search query (2-8 words) using the entity's proper name. "
            "Priority: entities from 'MENTIONED BUT NOT RETRIEVED'. "
            "Do not repeat previous_queries."
        )
    )


class QueryHop4GapSignature(dspy.Signature):
    """Identify and query for the last missing entity's Wikipedia article.

    Check BOTH sources for the missing entity:
    1. Entities from the CLAIM that are completely absent from retrieved_titles
    2. Entities MENTIONED in retrieved passages (from the summaries) whose OWN
       Wikipedia article title does NOT appear in retrieved_titles

    Generate a query using the missing entity's PROPER NAME — do not use descriptive
    queries. If the claim contains apparent typos or unusual spellings (e.g., a name
    that produced no results in prior hops), try the corrected or alternative spelling
    based on context from retrieved passages.

    Do NOT repeat any query already listed in previous_queries.
    Do NOT query for factual context already established — focus only on the entity
    whose own Wikipedia article is absent.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    retrieved_titles: str = dspy.InputField(
        desc="Comma-separated list of Wikipedia article titles already retrieved across previous hops"
    )
    previous_queries: str = dspy.InputField(
        desc="Semicolon-separated list of queries already used — do NOT repeat these"
    )
    summary: str = dspy.InputField(
        desc="Latest coverage report from summarization — check 'MENTIONED BUT NOT RETRIEVED' for entities needing direct lookup"
    )
    query: str = dspy.OutputField(
        desc=(
            "A direct, name-based query (2-6 words) for the missing entity's Wikipedia article. "
            "Use the entity's proper name (e.g., 'Allan Goldstein director', 'Adventist World magazine', "
            "'Texas Raiders aircraft', 'Paul Éluard poet'). "
            "Do not repeat previous_queries."
        )
    )


class QueryHop5FinalSignature(dspy.Signature):
    """Final targeted query to recover the last missing supporting document.

    You are given the FINAL LIST of 21 Wikipedia titles that will be returned as
    the output of the retrieval system. Compare these titles against the claim to
    find which key named entity still lacks its OWN Wikipedia article in the output.

    An entity is missing if:
    - The claim references it (as a person, film, show, song, place, company, or event)
    - Its own Wikipedia article title does NOT appear in retrieved_titles
    - Note: An entity MENTIONED INSIDE another article's text is NOT the same as
      having its own Wikipedia article retrieved — it still needs a direct search

    Generate a direct, name-based query for the single most important missing entity.
    Do NOT query for an entity already present by name in retrieved_titles.
    Do NOT repeat any query from previous_queries.
    """

    claim: str = dspy.InputField(desc="The factual claim being verified")
    retrieved_titles: str = dspy.InputField(
        desc="Comma-separated list of the 21 Wikipedia article titles currently in the final output"
    )
    previous_queries: str = dspy.InputField(
        desc="Semicolon-separated list of all queries used in previous hops — do NOT repeat these"
    )
    summary: str = dspy.InputField(
        desc="Latest coverage report — check 'MENTIONED BUT NOT RETRIEVED' for entities needing direct lookup"
    )
    query: str = dspy.OutputField(
        desc=(
            "A direct, name-based query (2-6 words) for the one most important missing entity's "
            "Wikipedia article. Use the entity's proper name (e.g., 'Sojourner Truth', "
            "'Shanghai Noon film', 'Ice Princess 2005 film', 'Jimi Hendrix'). "
            "Do not repeat previous_queries."
        )
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 22  # Increased from 15 for broader candidate coverage; max allowed is 25

        # Query generators with entity-focused signatures
        self.create_query_hop2 = dspy.ChainOfThought(QueryHop2Signature)
        self.create_query_hop3 = dspy.ChainOfThought(QueryHop3Signature)
        self.create_query_hop4 = dspy.ChainOfThought(QueryHop4GapSignature)
        self.create_query_hop5 = dspy.ChainOfThought(QueryHop5FinalSignature)

        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought(Summarize1Signature)
        self.summarize2 = dspy.ChainOfThought(Summarize2Signature)

    def _interleave_and_deduplicate(self, *hop_docs_lists, max_docs=21):
        """Round-robin interleave docs from multiple hops, deduplicate by title, take top max_docs.

        Interleaving ensures each hop contributes roughly equally to the final
        selection, preventing any single hop from dominating the 21-doc budget.
        With 4 hops, each contributes ~5 docs to the final 21.
        """
        seen = set()
        unique = []
        max_len = max((len(lst) for lst in hop_docs_lists), default=0)
        for i in range(max_len):
            for hop_docs in hop_docs_lists:
                if i < len(hop_docs):
                    doc = hop_docs[i]
                    title = doc.split(" | ")[0].strip().lower()
                    if title not in seen:
                        seen.add(title)
                        unique.append(doc)
                        if len(unique) >= max_docs:
                            return unique
        return unique

    def forward(self, claim):
        # HOP 1 - Initial retrieval with raw claim
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2 - Target a specific entity from the claim not yet fully retrieved.
        # Hop 1 used the raw claim as the query.
        hop2_query = self.create_query_hop2(
            claim=claim,
            summary_1=summary_1,
            previous_queries="[raw claim verbatim]",
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3 - Target remaining entity not covered by hops 1-2
        hop3_query = self.create_query_hop3(
            claim=claim,
            summary_1=summary_1,
            summary_2=summary_2,
            previous_queries=f"[raw claim]; {hop2_query}",
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # HOP 4 - Gap analysis: look at top retrieved titles, find what's still missing.
        # Use interleaved top-20 from hops 1-3 to get a balanced view of what's been retrieved.
        combined_so_far = self._interleave_and_deduplicate(
            hop1_docs, hop2_docs, hop3_docs, max_docs=30
        )
        retrieved_titles = ", ".join([d.split(" | ")[0] for d in combined_so_far[:20]])
        hop4_query = self.create_query_hop4(
            claim=claim,
            retrieved_titles=retrieved_titles,
            previous_queries=f"{hop2_query}; {hop3_query}",
            summary=summary_2,
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # HOP 5 - Final recovery hop: examine the actual preliminary final 21 output
        # and query for the entity that's still missing from it.
        # This fixes: (a) eviction — gold docs retrieved early but displaced by later hops,
        # (b) duplicate queries — hops 2/3/4 that duplicated each other wasted slots,
        # (c) summarizer hallucinations — hop4 skipped the right entity thinking it was covered.
        preliminary_final = self._interleave_and_deduplicate(
            hop1_docs, hop2_docs, hop3_docs, hop4_docs, max_docs=21
        )
        preliminary_titles = ", ".join([d.split(" | ")[0] for d in preliminary_final])
        hop5_query = self.create_query_hop5(
            claim=claim,
            retrieved_titles=preliminary_titles,
            previous_queries=f"{hop2_query}; {hop3_query}; {hop4_query}",
            summary=summary_2,
        ).query
        hop5_docs = self.retrieve_k(hop5_query).passages

        # "20 + 1 from hop5" merge strategy:
        # Keep the first 20 docs from the 4-hop round-robin unchanged (identical to the
        # existing 4-hop system for these slots, preserving all position-4 docs from
        # hops 2-4 that the full 5-hop round-robin incorrectly dropped).
        # Use hop5's top NEW result as the 21st slot only, replacing what would have been
        # hop1's 6th result in the pure 4-hop round-robin.
        # Fallback: if hop5 has no new doc, restore the natural 4-hop 21st slot.
        first_20 = preliminary_final[:20]
        first_20_titles = {d.split(" | ")[0].strip().lower() for d in first_20}

        hop5_insertion = None
        for doc in hop5_docs:
            title = doc.split(" | ")[0].strip().lower()
            if title not in first_20_titles:
                hop5_insertion = doc
                break

        if hop5_insertion is not None:
            final_docs = first_20 + [hop5_insertion]
        else:
            final_docs = preliminary_final  # fallback: natural 21st from 4-hop (hop1[5])

        return dspy.Prediction(retrieved_docs=final_docs)
