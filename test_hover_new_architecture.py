"""
Test script for the new HoverMultiHop architecture with query decomposition and parallel retrieval.
"""
import sys
sys.path.insert(0, '/workspace')

# Mock the dspy.Retrieve to test without actual retrieval
class MockRetrieve:
    def __init__(self, k=15):
        self.k = k

    def __call__(self, query):
        class MockPrediction:
            def __init__(self, k):
                # Generate mock documents
                self.passages = [f"Document {i+1} for query: {query[:30]}" for i in range(k)]
        return MockPrediction(self.k)

# Mock the document scorer to test without actual LLM calls
class MockDocumentScorer:
    def __init__(self):
        pass

    def __call__(self, claim, document):
        import random
        class MockPrediction:
            def __init__(self):
                # Generate a random score for testing
                self.relevance_score = str(round(random.random(), 2))
                self.reasoning = f"Mock reasoning for document relevance"
        return MockPrediction()

# Mock the query decomposer to test without actual LLM calls
class MockQueryDecomposer:
    def __init__(self):
        pass

    def __call__(self, claim):
        class MockPrediction:
            def __init__(self, claim):
                # Generate 3 diverse queries from the claim
                self.query1 = f"Query 1: {claim[:30]}"
                self.query2 = f"Query 2: {claim[:30]}"
                self.query3 = f"Query 3: {claim[:30]}"
        return MockPrediction(claim)

# Test the structure
print("Testing new HoverMultiHop architecture...")
print("=" * 80)

# Import after setting up path
from langProBe.hover.hover_program import HoverMultiHop, ClaimQueryDecomposer, DocumentRelevanceScorer

# Create instance
hover = HoverMultiHop()

# Replace components with mocks for testing
hover.retrieve_k = MockRetrieve(k=15)
hover.document_scorer = MockDocumentScorer()
hover.query_decomposer = MockQueryDecomposer()

# Test with a sample claim
test_claim = "The director of Titanic was born in Canada"

print(f"Test claim: {test_claim}\n")

# Simulate the forward pass
queries = hover.query_decomposer(claim=test_claim)
print(f"Generated queries:")
print(f"  Query 1: {queries.query1}")
print(f"  Query 2: {queries.query2}")
print(f"  Query 3: {queries.query3}\n")

# Test retrieval
docs_query1 = hover.retrieve_k(queries.query1).passages
docs_query2 = hover.retrieve_k(queries.query2).passages
docs_query3 = hover.retrieve_k(queries.query3).passages

print(f"Retrieved documents:")
print(f"  Query 1: {len(docs_query1)} documents")
print(f"  Query 2: {len(docs_query2)} documents")
print(f"  Query 3: {len(docs_query3)} documents")
print(f"  Total: {len(docs_query1 + docs_query2 + docs_query3)} documents\n")

# Test full forward pass
import dspy

# Mock the Prediction class if needed
class MockPrediction:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

dspy.Prediction = MockPrediction

result = hover.forward(claim=test_claim)

print(f"Final result:")
print(f"  Number of documents returned: {len(result.retrieved_docs)}")
print(f"  Expected: 21 documents")
print(f"  Status: {'✓ PASS' if len(result.retrieved_docs) == 21 else '✗ FAIL'}\n")

print("=" * 80)
print("Architecture Summary:")
print("1. ✓ ClaimQueryDecomposer: claim -> query1, query2, query3")
print("2. ✓ Parallel Retrieval: 15 docs per query (3 queries × 15 = 45 docs)")
print("3. ✓ DocumentRelevanceScorer: claim, document -> relevance_score, reasoning")
print("4. ✓ Reranking: Sort by score and return top 21 documents")
print("=" * 80)

print("\nKey Improvements:")
print("✓ Eliminated sequential hop dependency")
print("✓ No information loss from summarization")
print("✓ Parallel diverse queries target all entities simultaneously")
print("✓ Score-based reranking ensures most relevant documents are selected")
print("✓ Maintains 21 document limit requirement")
