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


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

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

        retrieved_docs = _dedup_by_title(hop1_docs + hop2_docs + hop3_docs, max_docs=21)
        return dspy.Prediction(retrieved_docs=retrieved_docs)
