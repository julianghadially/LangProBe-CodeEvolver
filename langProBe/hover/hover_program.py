import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Query expansion to generate complementary search queries
        self.query_expander = dspy.ChainOfThought("claim->queries")

        # Retrieval modules with different k values
        self.retrieve_21 = dspy.Retrieve(k=21)
        self.retrieve_14 = dspy.Retrieve(k=14)

        # Document relevance scorer
        self.doc_scorer = dspy.ChainOfThought("claim,document->relevance_score")

    def forward(self, claim):
        # STEP 1: Generate 2 complementary search queries from the claim
        expanded_queries = self.query_expander(claim=claim).queries

        # Parse the queries (assuming format with 2 queries)
        # The output should contain 2 queries - extract them
        query_list = []
        if isinstance(expanded_queries, str):
            # Split by newlines or numbered format
            lines = [line.strip() for line in expanded_queries.strip().split('\n') if line.strip()]
            for line in lines:
                # Remove numbering like "1.", "2." if present
                cleaned = line.lstrip('0123456789.-) ').strip()
                if cleaned:
                    query_list.append(cleaned)

        # Ensure we have exactly 2 queries, fallback to claim if needed
        if len(query_list) < 2:
            query_list = [claim, claim]
        query1 = query_list[0]
        query2 = query_list[1] if len(query_list) > 1 else query_list[0]

        # STEP 2: Retrieve documents for both expanded queries
        # Query 1: retrieve k=21 documents
        hop1_docs = self.retrieve_21(query1).passages

        # Query 2: retrieve k=14 documents
        hop2_docs = self.retrieve_14(query2).passages

        # STEP 3: Deduplicate documents by title
        seen_titles = set()
        unique_docs = []

        for doc in hop1_docs + hop2_docs:
            # Extract title (assuming format "title | text" or just using first part)
            doc_title = doc.split('|')[0].strip() if '|' in doc else doc[:100]

            if doc_title not in seen_titles:
                seen_titles.add(doc_title)
                unique_docs.append(doc)

        # STEP 4: Score all unique documents
        scored_docs = []
        for doc in unique_docs:
            score_result = self.doc_scorer(claim=claim, document=doc).relevance_score

            # Parse score (extract number from response)
            try:
                # Try to extract numeric score
                score_str = ''.join(filter(lambda x: x.isdigit() or x == '.', str(score_result)))
                score = float(score_str) if score_str else 0.0
            except:
                score = 0.0

            scored_docs.append((score, doc))

        # STEP 5: Sort by score (descending) and select top 21
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_21_docs = [doc for score, doc in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
