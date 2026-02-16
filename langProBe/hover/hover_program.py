import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


def deduplicate_passages(passages):
    """Deduplicate passages based on document title.

    Args:
        passages: List of passages in "Document | Title..." format

    Returns:
        List of unique passages (first occurrence preserved)
    """
    seen_titles = set()
    unique_passages = []

    for passage in passages:
        # Extract title from "Document | Title..." format
        title = passage.split(" | ")[0] if " | " in passage else passage

        # Normalize title for comparison
        normalized_title = dspy.evaluate.normalize_text(title)

        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_passages.append(passage)

    return unique_passages


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        # Deduplicate hop1 results
        hop1_docs_unique = deduplicate_passages(hop1_docs)
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs_unique
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        # Deduplicate hop2 results and remove any that already appeared in hop1
        hop2_docs_unique = deduplicate_passages(hop1_docs_unique + hop2_docs)
        # Keep only new documents from hop2
        hop2_docs_new = hop2_docs_unique[len(hop1_docs_unique):]
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs_new
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages
        # Deduplicate hop3 results and remove any that already appeared in hop1 or hop2
        all_previous_docs = hop1_docs_unique + hop2_docs_new
        hop3_docs_unique = deduplicate_passages(all_previous_docs + hop3_docs)
        # Keep only new documents from hop3
        hop3_docs_new = hop3_docs_unique[len(all_previous_docs):]

        # Aggregate all unique documents using set-based approach
        # The deduplicate_passages function ensures no duplicates in final output
        final_docs = deduplicate_passages(hop1_docs_unique + hop2_docs_new + hop3_docs_new)

        return dspy.Prediction(retrieved_docs=final_docs)
