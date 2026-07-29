import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 12
        self.max_docs = 21
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _interleave_dedup(self, hop_doc_lists, max_docs):
        """Round-robin interleave per-hop ranked lists, dedup by title, cap."""
        seen_titles = set()
        merged = []
        max_len = max((len(h) for h in hop_doc_lists), default=0)
        for i in range(max_len):
            for hop_docs in hop_doc_lists:
                if i < len(hop_docs):
                    doc = hop_docs[i]
                    title = doc.split(" | ")[0]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        merged.append(doc)
                        if len(merged) >= max_docs:
                            return merged
        return merged

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

        merged = self._interleave_dedup(
            [hop1_docs, hop2_docs, hop3_docs], self.max_docs
        )
        return dspy.Prediction(retrieved_docs=merged)
