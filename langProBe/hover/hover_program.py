import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyNextTarget(dspy.Signature):
    """You are retrieving Wikipedia articles to support verification of a multi-hop factual claim.

    Given the claim and Wikipedia passages already retrieved, identify the SINGLE most important
    Wikipedia article still needed to verify the claim.

    Steps:
    1. List ALL named entities explicitly mentioned in the claim (people, places, organizations,
       works, songs, films, awards, etc.)
    2. For EACH named entity from step 1, check if its own Wikipedia article appears in the
       retrieved_passages titles (the text before the " | " separator in each passage).
    3. Output the FIRST named entity from step 1 that does NOT yet have its own article retrieved.
    4. If ALL named entities in the claim already have their own article retrieved, scan the
       retrieved passage TEXT for implied entities not yet retrieved (e.g., the company that
       produced a film, the co-winner of an award, the subject of a biography) — output the most
       important one not yet retrieved.
    5. CRITICAL: Before finalizing, check previous_queries. You MUST NOT output any entity name
       that appears in previous_queries. If your best candidate matches a previous query,
       choose the NEXT best uncovered entity instead.

    Output ONLY a concise Wikipedia article title or entity name — nothing else.
    Good examples: "Pablo Escobar", "Apple Inc.", "Gene Kelly", "Sheldon Lee Glashow"
    Bad examples: "Who starred in Narcos?", "Was Steven Weinberg a professor?"
    Do NOT output a question. Do NOT output a sentence. Output a Wikipedia title or entity name only.
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(
        desc="Wikipedia passages already retrieved (format: 'ArticleTitle | text excerpt...')"
    )
    previous_queries: str = dspy.InputField(
        desc="Entity names/queries already issued in previous hops, separated by '; '. "
             "You MUST NOT repeat any of these — output a DIFFERENT entity name."
    )
    query: str = dspy.OutputField(
        desc="A single Wikipedia article title or entity name to search for next — "
             "NOT a question, NOT a sentence, NOT a repeat of anything in previous_queries"
    )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10  # Increased from 7 for better unique-doc coverage
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.identify_hop2_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop3_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop4_target = dspy.ChainOfThought(IdentifyNextTarget)

    def forward(self, claim):
        # HOP 1: Direct retrieval on raw claim
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Identify the next most important missing entity from the claim
        hop2_query = self.identify_hop2_target(
            claim=claim,
            retrieved_passages="\n---\n".join(hop1_docs),
            previous_queries="(none yet)"
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Identify another missing entity, aware of hop 2's query
        hop3_query = self.identify_hop3_target(
            claim=claim,
            retrieved_passages="\n---\n".join(hop1_docs + hop2_docs),
            previous_queries=hop2_query
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # HOP 4: Final targeted sweep for any remaining uncovered entity
        hop4_query = self.identify_hop4_target(
            claim=claim,
            retrieved_passages="\n---\n".join(hop1_docs + hop2_docs + hop3_docs),
            previous_queries=f"{hop2_query}; {hop3_query}"
        ).query
        hop4_docs = self.retrieve_k(hop4_query).passages

        # Combine all docs, deduplicate by article title (first occurrence wins)
        all_docs = hop1_docs + hop2_docs + hop3_docs + hop4_docs
        seen_titles = set()
        unique_docs = []
        for doc in all_docs:
            title = doc.split(" | ")[0].strip().lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

        return dspy.Prediction(retrieved_docs=unique_docs[:21])
