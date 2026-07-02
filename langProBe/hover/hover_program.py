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
        # GAP HOP: identify Wikipedia articles for entities named in the claim/summaries
        # but whose own article title is not yet among the retrieved titles, then fetch them.
        self.identify_missing = dspy.ChainOfThought(
            "claim, summaries, retrieved_titles -> missing_titles"
        )

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

        # GAP HOP: fetch dedicated articles for named entities not yet retrieved.
        unique_titles = {
            self._doc_title(p)
            for hop_docs in (hop1_docs, hop2_docs, hop3_docs)
            for p in hop_docs
        }
        summaries = summary_1 + "\n" + summary_2
        missing_titles_str = self.identify_missing(
            claim=claim,
            summaries=summaries,
            retrieved_titles=", ".join(sorted(unique_titles)),
        ).missing_titles
        missing_titles = []
        for raw in missing_titles_str.replace("\n", ",").split(","):
            title = raw.strip()
            if title and title not in unique_titles:
                unique_titles.add(title)
                missing_titles.append(title)
                if len(missing_titles) >= 3:
                    break
        gap_docs_list = [
            self.retrieve_k(title).passages for title in missing_titles
        ]

        return dspy.Prediction(
            retrieved_docs=self._round_robin_dedup(
                [hop1_docs, hop2_docs, hop3_docs] + gap_docs_list
            )
        )
