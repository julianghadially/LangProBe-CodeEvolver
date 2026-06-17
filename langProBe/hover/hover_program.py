import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class IdentifyNextTarget(dspy.Signature):
    """You are retrieving Wikipedia articles to support verification of a multi-hop factual claim.

    Given the claim and Wikipedia passages already retrieved, identify the single most important
    Wikipedia article still needed to verify the claim.

    Steps:
    1. List all named entities in the claim (people, places, organizations, works, titles)
    2. Check which named entities are already covered by the retrieved passage titles
    3. Also scan the retrieved passage TEXT for any new named entities the claim implies but haven't been retrieved
    4. Pick the single most important uncovered entity/article

    Output ONLY a concise Wikipedia article title or entity name.
    Good examples: "Pablo Escobar", "Worldview Entertainment", "Ibn Tufail", "Gene Kelly"
    Bad examples: "Who wrote The Four-Chambered Heart?", "Is Bob Fosse born in 1912?"
    Do NOT output a question. Do NOT output a sentence. Output a Wikipedia title or entity name only.
    """
    claim: str = dspy.InputField(desc="The factual claim to verify")
    retrieved_passages: str = dspy.InputField(desc="Wikipedia passages already retrieved (format: 'ArticleTitle | text excerpt...')")
    query: str = dspy.OutputField(desc="A Wikipedia article title or entity name to search for next — not a question, not a sentence")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.identify_hop2_target = dspy.ChainOfThought(IdentifyNextTarget)
        self.identify_hop3_target = dspy.ChainOfThought(IdentifyNextTarget)

    def forward(self, claim):
        # HOP 1: Direct retrieval on raw claim
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Identify the next most important missing entity from the claim
        hop2_query = self.identify_hop2_target(
            claim=claim,
            retrieved_passages="\n---\n".join(hop1_docs)
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Identify another missing entity (aware of what hop 1 and 2 retrieved)
        hop3_query = self.identify_hop3_target(
            claim=claim,
            retrieved_passages="\n---\n".join(hop1_docs + hop2_docs)
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all docs, deduplicate by article title (first occurrence wins)
        all_docs = hop1_docs + hop2_docs + hop3_docs
        seen_titles = set()
        unique_docs = []
        for doc in all_docs:
            title = doc.split(" | ")[0].strip().lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

        return dspy.Prediction(retrieved_docs=unique_docs[:21])
