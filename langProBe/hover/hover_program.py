import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 15
        self.MAX_RETRIEVED_DOCS = 21
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.rerank = dspy.ChainOfThought("claim,retrieved_docs->ranked_docs")

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

        # Combine all retrieved documents (45 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Rerank and select top 21 most relevant documents
        ranked_docs = self.rerank(
            claim=claim, retrieved_docs=all_docs
        ).ranked_docs

        # Ensure we return exactly MAX_RETRIEVED_DOCS (21) documents
        if isinstance(ranked_docs, list):
            final_docs = ranked_docs[:self.MAX_RETRIEVED_DOCS]
        else:
            # If reranker returns a string or other format, fall back to original docs
            final_docs = all_docs[:self.MAX_RETRIEVED_DOCS]

        return dspy.Prediction(retrieved_docs=final_docs)


