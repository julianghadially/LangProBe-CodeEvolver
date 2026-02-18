import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10
        self.create_query_hop2 = dspy.ChainOfThought("claim,entities_from_hop1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,entities_from_hop1,entities_from_hop2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->entities")
        self.summarize2 = dspy.ChainOfThought("claim,passages->entities")

    def forward(self, claim):
        # HOP 1 - Retrieve 10 documents
        hop1_docs = self.retrieve_k(claim).passages
        entities_from_hop1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).entities  # Extract key entities and document titles from top k docs

        # HOP 2 - Retrieve 10 documents
        hop2_query = self.create_query_hop2(claim=claim, entities_from_hop1=entities_from_hop1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        entities_from_hop2 = self.summarize2(
            claim=claim, passages=hop2_docs
        ).entities  # Extract key entities and document titles from hop 2 docs

        # HOP 3 - Retrieve 10 documents
        hop3_query = self.create_query_hop3(
            claim=claim, entities_from_hop1=entities_from_hop1, entities_from_hop2=entities_from_hop2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all documents (30 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate by document title, keeping first occurrence
        seen_titles = set()
        unique_docs = []
        for doc in all_docs:
            # Extract title from document (assuming doc has 'long_text' attribute with title)
            # DSPy passages typically have format with title at the start
            doc_text = str(doc)
            # Try to extract title (typically first line or before first newline)
            title = doc_text.split('\n')[0] if '\n' in doc_text else doc_text[:100]

            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)

                # Stop once we have 21 unique documents
                if len(unique_docs) == 21:
                    break

        return dspy.Prediction(retrieved_docs=unique_docs)
