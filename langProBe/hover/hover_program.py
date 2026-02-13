import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # Track titles from hop 1
        hop1_titles = set()
        for doc in hop1_docs:
            title = doc.split(' | ')[0]
            hop1_titles.add(title)

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # Filter out duplicates from hop 2
        filtered_hop2_docs = []
        hop2_titles = set()
        for doc in hop2_docs:
            title = doc.split(' | ')[0]
            if title not in hop1_titles:
                filtered_hop2_docs.append(doc)
                hop2_titles.add(title)

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=filtered_hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Filter out duplicates from hop 3
        all_previous_titles = hop1_titles | hop2_titles
        filtered_hop3_docs = []
        for doc in hop3_docs:
            title = doc.split(' | ')[0]
            if title not in all_previous_titles:
                filtered_hop3_docs.append(doc)

        return dspy.Prediction(retrieved_docs=hop1_docs + filtered_hop2_docs + filtered_hop3_docs)


