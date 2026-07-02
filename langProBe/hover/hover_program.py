import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

MAX_DOCS = 21


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Retrieve more candidates per hop (k=10) than the final 21 budget, then dedup by
        # title and pick the best 21 via round-robin. This trades no extra searches (still 3,
        # so the resource penalty is unchanged) for a larger unique-document candidate pool,
        # reclaiming slots previously wasted on duplicate passages across hops.
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    @staticmethod
    def _doc_title(passage):
        return passage.split(" | ")[0]

    def _round_robin_dedup(self, hop_docs):
        """Round-robin across hops (each ColBERT-ranked), dedup by title, cap at MAX_DOCS.

        Balanced representation per hop avoids one hop monopolising the budget, so refined
        follow-up hops get their fair share of the 21 slots allotted to search candidates."""
        seen = set()
        docs = []
        max_len = max((len(h) for h in hop_docs), default=0)
        for i in range(max_len):
            for hop in hop_docs:
                if i < len(hop):
                    title = self._doc_title(hop[i])
                    if title not in seen:
                        seen.add(title)
                        docs.append(hop[i])
                        if len(docs) >= MAX_DOCS:
                            return docs
        return docs[:MAX_DOCS]

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

        return dspy.Prediction(
            retrieved_docs=self._round_robin_dedup([hop1_docs, hop2_docs, hop3_docs])
        )
