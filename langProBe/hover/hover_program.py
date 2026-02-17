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


class DocumentRelevanceScorerSignature(dspy.Signature):
    """Score how relevant a document is to answering or verifying the given claim.
    Provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant),
    along with reasoning explaining why this score was assigned."""

    claim = dspy.InputField(desc="The claim to verify or answer")
    document = dspy.InputField(desc="The document to score for relevance")
    relevance_score = dspy.OutputField(desc="Relevance score between 0.0 and 1.0")
    reasoning = dspy.OutputField(desc="Explanation of why this score was assigned")


class DocumentRelevanceScorer(dspy.Module):
    """DSPy module that scores document relevance to a claim."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(DocumentRelevanceScorerSignature)

    def forward(self, claim, document):
        result = self.score(claim=claim, document=document)
        return dspy.Prediction(
            relevance_score=result.relevance_score,
            reasoning=result.reasoning
        )


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using query decomposition and parallel retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Uses query decomposition to generate 3 diverse queries targeting different aspects of the claim
    - Retrieves 15 documents per query in parallel (total 45 documents)
    - Scores all 45 documents for relevance to the claim
    - Reranks documents by relevance score and returns top 21'''

    def __init__(self):
        super().__init__()
        self.k = 15  # Retrieve 15 documents per query
        self.query_decomposer = ClaimQueryDecomposer()
        self.document_scorer = DocumentRelevanceScorer()
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Step 1: Decompose claim into 3 diverse queries
        queries = self.query_decomposer(claim=claim)

        # Step 2: Parallel retrieval - retrieve k=15 documents per query (total 45 documents)
        docs_query1 = self.retrieve_k(queries.query1).passages
        docs_query2 = self.retrieve_k(queries.query2).passages
        docs_query3 = self.retrieve_k(queries.query3).passages

        # Combine all retrieved documents
        all_docs = docs_query1 + docs_query2 + docs_query3

        # Step 3: Score each document for relevance to the claim
        scored_docs = []
        for doc in all_docs:
            score_result = self.document_scorer(claim=claim, document=doc)
            try:
                # Try to parse the relevance_score as a float
                score = float(score_result.relevance_score)
            except (ValueError, TypeError):
                # If parsing fails, assign a default score of 0.5
                score = 0.5

            scored_docs.append({
                'document': doc,
                'score': score,
                'reasoning': score_result.reasoning
            })

        # Step 4: Rerank documents by relevance score (descending order)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # Step 5: Return top 21 documents
        top_21_docs = [item['document'] for item in scored_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
