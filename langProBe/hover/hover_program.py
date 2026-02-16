import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class RerankPassages(dspy.Signature):
    """Evaluate the relevance of a passage to a given claim."""
    claim = dspy.InputField()
    passage = dspy.InputField()
    relevance_score = dspy.OutputField(desc="relevance score from 0 (irrelevant) to 10 (highly relevant)")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15  # Increased from 7 to 15 for 45 total candidates
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.reranker = dspy.ChainOfThought(RerankPassages)

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

        # Pool all candidate documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate by normalized title
        seen_titles = {}
        unique_docs = []
        for doc in all_docs:
            # Extract title - assuming doc format is "title | content" or similar
            # Normalize by converting to lowercase and stripping whitespace
            doc_str = str(doc)
            title = doc_str.split('|')[0].strip().lower() if '|' in doc_str else doc_str[:100].strip().lower()

            if title not in seen_titles:
                seen_titles[title] = True
                unique_docs.append(doc)

        # Rerank each unique document
        scored_docs = []
        for doc in unique_docs:
            result = self.reranker(claim=claim, passage=str(doc))
            try:
                # Extract numeric score from the relevance_score field
                score = int(result.relevance_score)
            except (ValueError, AttributeError):
                # If score is not a valid integer, try to extract it from the string
                try:
                    score_str = str(result.relevance_score).strip()
                    # Extract first number found in the string
                    import re
                    match = re.search(r'\d+', score_str)
                    score = int(match.group()) if match else 0
                except:
                    score = 0

            scored_docs.append((score, doc))

        # Sort by relevance score descending and take top 21
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_21_docs = [doc for score, doc in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
