import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        self.extract_key_terms = dspy.Predict("claim->key_terms")
        self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")
        self.create_query_hop3 = dspy.Predict("claim,key_terms,hop1_titles,hop2_titles->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Extract key terms from claim (3-5 key named entities or concepts)
        key_terms = self.extract_key_terms(claim=claim).key_terms

        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100] for doc in hop1_docs])

        # Track seen documents using normalized titles
        seen_titles = set()
        final_docs = []

        # Add hop1 docs and track their titles
        for doc in hop1_docs:
            title = doc.split("\n")[0] if "\n" in doc else doc[:100]
            seen_titles.add(title)
            final_docs.append(doc)

        # HOP 2
        hop2_query = self.create_query_hop2(
            claim=claim, key_terms=key_terms, hop1_titles=hop1_titles
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100] for doc in hop2_docs])

        # Add hop2 docs, filtering duplicates
        for doc in hop2_docs:
            title = doc.split("\n")[0] if "\n" in doc else doc[:100]
            if title not in seen_titles:
                seen_titles.add(title)
                final_docs.append(doc)

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, key_terms=key_terms, hop1_titles=hop1_titles, hop2_titles=hop2_titles
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Add hop3 docs, filtering duplicates
        for doc in hop3_docs:
            title = doc.split("\n")[0] if "\n" in doc else doc[:100]
            if title not in seen_titles:
                seen_titles.add(title)
                final_docs.append(doc)

        return dspy.Prediction(retrieved_docs=final_docs)


