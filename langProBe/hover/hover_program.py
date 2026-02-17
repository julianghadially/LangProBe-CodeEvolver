import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimQueryDecomposerSignature(dspy.Signature):
    """Generate 3 diverse search queries that focus on different entities and aspects mentioned in the claim.
    Each query should target a specific entity, concept, or relationship in the claim to ensure comprehensive coverage.
    For example, for a claim like 'The director of Film X was born in Country Y', generate:
    - query1: Focus on the film and its director
    - query2: Focus on the film itself
    - query3: Focus on birthplace information"""

    claim = dspy.InputField(desc="The claim to decompose into search queries")
    query1 = dspy.OutputField(desc="First search query focusing on a specific entity or aspect")
    query2 = dspy.OutputField(desc="Second search query focusing on a different entity or aspect")
    query3 = dspy.OutputField(desc="Third search query focusing on yet another entity or aspect")


class ClaimQueryDecomposer(dspy.Module):
    """DSPy module that decomposes a claim into 3 diverse search queries."""

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(ClaimQueryDecomposerSignature)

    def forward(self, claim):
        result = self.decompose(claim=claim)
        return dspy.Prediction(
            query1=result.query1,
            query2=result.query2,
            query3=result.query3
        )


def keyword_rerank_documents(claim, documents, top_k=21):
    """Rerank documents based on keyword overlap with the claim.

    This function:
    1. Deduplicates documents by title
    2. Scores each document by counting overlapping tokens between the claim and document (title + content)
    3. Returns top_k documents sorted by overlap score

    Args:
        claim: The claim text to match against
        documents: List of document strings in format "Title | Content"
        top_k: Number of top documents to return (default: 21)

    Returns:
        List of top_k documents with highest overlap scores
    """
    import dspy.evaluate

    # Normalize the claim text
    claim_normalized = dspy.evaluate.normalize_text(claim)
    claim_tokens = set(claim_normalized.split())

    # Deduplicate documents by title
    seen_titles = {}
    for doc in documents:
        # Extract title (text before " | ")
        if " | " in doc:
            title = doc.split(" | ")[0]
        else:
            title = doc

        # Keep first occurrence of each unique title
        title_normalized = dspy.evaluate.normalize_text(title)
        if title_normalized not in seen_titles:
            seen_titles[title_normalized] = doc

    # Score each unique document by keyword overlap
    scored_docs = []
    for title_normalized, doc in seen_titles.items():
        # Normalize the entire document text
        doc_normalized = dspy.evaluate.normalize_text(doc)
        doc_tokens = set(doc_normalized.split())

        # Count overlapping tokens
        overlap_count = len(claim_tokens & doc_tokens)

        scored_docs.append({
            'document': doc,
            'score': overlap_count
        })

    # Sort by overlap score (descending) and return top_k
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    return [item['document'] for item in scored_docs[:top_k]]


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using query decomposition and parallel retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Uses query decomposition to generate 3 diverse queries targeting different aspects of the claim
    - Retrieves 21 documents per query in parallel (total 63 documents)
    - Deduplicates documents by title to remove redundant entries
    - Reranks documents using lightweight keyword-based scoring (token overlap between claim and document)
    - Returns top 21 unique documents with highest overlap scores

    This approach eliminates expensive LLM-based relevance scoring while better capturing documents
    that mention specific entities from the claim.'''

    def __init__(self):
        super().__init__()
        self.k = 21  # Retrieve 21 documents per query
        self.query_decomposer = ClaimQueryDecomposer()
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Step 1: Decompose claim into 3 diverse queries
        queries = self.query_decomposer(claim=claim)

        # Step 2: Parallel retrieval - retrieve k=21 documents per query (total 63 documents)
        docs_query1 = self.retrieve_k(queries.query1).passages
        docs_query2 = self.retrieve_k(queries.query2).passages
        docs_query3 = self.retrieve_k(queries.query3).passages

        # Combine all retrieved documents
        all_docs = docs_query1 + docs_query2 + docs_query3

        # Step 3: Deduplicate and rerank documents using keyword-based overlap scoring
        top_21_docs = keyword_rerank_documents(claim, all_docs, top_k=21)

        return dspy.Prediction(retrieved_docs=top_21_docs)
