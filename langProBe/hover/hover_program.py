import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15
        self.create_query_hop2 = dspy.ChainOfThought("claim->alternative_query")
        self.create_query_hop3 = dspy.ChainOfThought("claim->entity_focused_query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.relevance_scorer = dspy.ChainOfThought("claim,passages->relevant_titles")

    def forward(self, claim):
        # HOP 1: Retrieve using original claim
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Generate alternative query phrasing
        hop2_query = self.create_query_hop2(claim=claim).alternative_query
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Generate entity-focused query
        hop3_query = self.create_query_hop3(claim=claim).entity_focused_query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all 45 retrieved documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # LM-based relevance scoring: get top 21 most relevant document titles
        relevant_titles_output = self.relevance_scorer(
            claim=claim, passages=all_docs
        ).relevant_titles

        # Parse comma-separated titles
        relevant_titles = [title.strip() for title in relevant_titles_output.split(',')]

        # Rerank: filter documents to keep only those in the relevant_titles list
        # Preserve original document order
        reranked_docs = []
        for doc in all_docs:
            # Extract title from document (assuming doc has a 'title' attribute or is a dict)
            doc_title = doc.get('title') if isinstance(doc, dict) else getattr(doc, 'title', str(doc))
            if doc_title in relevant_titles and len(reranked_docs) < 21:
                reranked_docs.append(doc)

        return dspy.Prediction(retrieved_docs=reranked_docs)
