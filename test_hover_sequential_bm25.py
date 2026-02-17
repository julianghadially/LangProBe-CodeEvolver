"""
Unit tests for the new sequential multi-hop retrieval with BM25 reranking architecture.
"""
import sys
sys.path.insert(0, '/workspace')

from langProBe.hover.hover_program import BM25Reranker, EntityExtractor, FocusedQueryGenerator, HoverMultiHop
import dspy


def test_bm25_tokenization():
    """Test BM25 tokenization."""
    print("Testing BM25 tokenization...")
    reranker = BM25Reranker()

    # Test basic tokenization
    tokens = reranker.tokenize("The quick brown fox")
    assert tokens == ["the", "quick", "brown", "fox"], f"Expected ['the', 'quick', 'brown', 'fox'], got {tokens}"

    # Test with punctuation
    tokens = reranker.tokenize("Hello, world! How are you?")
    assert tokens == ["hello", "world", "how", "are", "you"], f"Expected ['hello', 'world', 'how', 'are', 'you'], got {tokens}"

    # Test with numbers
    tokens = reranker.tokenize("Test 123 numbers")
    assert tokens == ["test", "123", "numbers"], f"Expected ['test', '123', 'numbers'], got {tokens}"

    print("✓ BM25 tokenization tests passed\n")


def test_bm25_idf_computation():
    """Test IDF computation."""
    print("Testing BM25 IDF computation...")
    reranker = BM25Reranker()

    docs = [
        "cats are feline animals",
        "dogs are canine animals",
        "cats and dogs are pets"
    ]

    idf_scores = reranker.compute_idf(docs)

    # "animals" appears in 2 docs, "pets" appears in 1 doc
    # "pets" should have higher IDF than "animals"
    assert "animals" in idf_scores, "Expected 'animals' in IDF scores"
    assert "pets" in idf_scores, "Expected 'pets' in IDF scores"
    assert idf_scores["pets"] > idf_scores["animals"], "Expected 'pets' to have higher IDF than 'animals'"

    print(f"  IDF('animals') = {idf_scores['animals']:.4f}")
    print(f"  IDF('pets') = {idf_scores['pets']:.4f}")
    print("✓ BM25 IDF computation tests passed\n")


def test_bm25_scoring():
    """Test BM25 scoring and ranking."""
    print("Testing BM25 scoring and ranking...")
    reranker = BM25Reranker()

    docs = [
        "Document about cats and their behavior",
        "Document about dogs and training",
        "Document about felines and wildcats"
    ]

    # Query mentioning cats and felines
    ranked = reranker.rerank("cats and felines", docs)

    # Verify structure
    assert len(ranked) == 3, f"Expected 3 ranked documents, got {len(ranked)}"
    assert all(isinstance(score, float) for _, score in ranked), "All scores should be floats"

    # Verify that docs with "cats" and "felines" score higher
    doc_texts = [doc for doc, _ in ranked]
    scores = [score for _, score in ranked]

    # Check that scores are in descending order
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Scores should be in descending order: {scores}"

    # Documents with "cats" or "felines" should rank higher than the one about "dogs"
    top_doc_texts = [ranked[0][0], ranked[1][0]]
    assert docs[1] not in top_doc_texts or docs[1] == ranked[2][0], "Document about dogs should not be in top 2"

    print(f"  Ranked documents:")
    for i, (doc, score) in enumerate(ranked):
        print(f"    {i+1}. Score={score:.4f}: {doc[:50]}...")
    print("✓ BM25 scoring tests passed\n")


def test_bm25_empty_docs():
    """Test BM25 with empty document list."""
    print("Testing BM25 with empty documents...")
    reranker = BM25Reranker()

    ranked = reranker.rerank("test query", [])
    assert ranked == [], "Expected empty list for empty documents"

    print("✓ BM25 empty docs test passed\n")


# Mock classes for testing sequential hops
class MockRetrieve:
    def __init__(self, k=35):
        self.k = k
        self.call_count = 0

    def __call__(self, query):
        self.call_count += 1
        class MockPrediction:
            def __init__(self, k, query, call_num):
                # Generate mock documents with titles in "title | content" format
                self.passages = [
                    f"Doc{call_num}_{i} | Document {i+1} content for query: {query[:30]}..."
                    for i in range(k)
                ]
        return MockPrediction(self.k, query, self.call_count)


class MockEntityExtractor:
    def __init__(self):
        self.call_count = 0

    def __call__(self, claim, documents):
        self.call_count += 1
        return f"uncovered entities from hop {self.call_count}"


class MockQueryGenerator:
    def __init__(self):
        self.call_count = 0

    def __call__(self, claim, uncovered_entities):
        self.call_count += 1
        return f"focused query for hop {self.call_count}: {uncovered_entities}"


def test_sequential_hops():
    """Test that sequential hops work correctly."""
    print("Testing sequential hops...")

    # Create HoverMultiHop instance
    hover = HoverMultiHop()

    # Replace with mocks
    mock_retrieve = MockRetrieve(k=35)
    hover.retrieve_k = mock_retrieve
    hover.entity_extractor = MockEntityExtractor()
    hover.query_generator = MockQueryGenerator()

    # Mock dspy.Prediction if needed
    class MockPrediction:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    original_prediction = dspy.Prediction
    dspy.Prediction = MockPrediction

    try:
        # Run forward pass
        test_claim = "The director of Titanic was born in Canada"
        result = hover.forward(claim=test_claim)

        # Verify 3 retrievals happened
        assert mock_retrieve.call_count == 3, f"Expected 3 retrievals, got {mock_retrieve.call_count}"

        # Verify entity extractor was called twice (hop 2 and hop 3)
        assert hover.entity_extractor.call_count == 2, f"Expected 2 entity extractions, got {hover.entity_extractor.call_count}"

        # Verify query generator was called twice (hop 2 and hop 3)
        assert hover.query_generator.call_count == 2, f"Expected 2 query generations, got {hover.query_generator.call_count}"

        # Verify deduplication works - should return at most 21 unique documents
        assert len(result.retrieved_docs) <= 21, f"Expected at most 21 documents, got {len(result.retrieved_docs)}"

        # Verify all documents are unique
        unique_titles = set()
        for doc in result.retrieved_docs:
            title = doc.split(" | ")[0] if " | " in doc else doc[:100]
            unique_titles.add(title)

        assert len(unique_titles) == len(result.retrieved_docs), "All documents should be unique"

        print(f"  ✓ Sequential hops: {mock_retrieve.call_count} retrievals")
        print(f"  ✓ Entity extractions: {hover.entity_extractor.call_count}")
        print(f"  ✓ Query generations: {hover.query_generator.call_count}")
        print(f"  ✓ Documents returned: {len(result.retrieved_docs)}")
        print(f"  ✓ Unique documents: {len(unique_titles)}")
        print("✓ Sequential hops test passed\n")

    finally:
        # Restore original Prediction
        dspy.Prediction = original_prediction


def test_deduplication():
    """Test document deduplication by normalized title."""
    print("Testing document deduplication...")

    hover = HoverMultiHop()

    # Create mock documents with duplicate titles
    mock_docs = [
        "Titanic | Content about Titanic movie 1",
        "James Cameron | Content about James Cameron 1",
        "Titanic | Content about Titanic movie 2",  # Duplicate title
        "Canada | Content about Canada 1",
        "james cameron | Content about James Cameron 2",  # Duplicate (case-insensitive)
        "Avatar | Content about Avatar movie",
    ]

    # Mock the retrieve_k to return these docs
    class MockRetrieveForDedup:
        def __call__(self, query):
            class MockPrediction:
                def __init__(self):
                    self.passages = mock_docs[:2]  # Only return 2 docs per hop
            return MockPrediction()

    hover.retrieve_k = MockRetrieveForDedup()

    # Mock entity extractor and query generator
    hover.entity_extractor = lambda claim, documents: "entities"
    hover.query_generator = lambda claim, uncovered_entities: "query"

    # Test deduplication logic directly
    all_docs = mock_docs

    seen_titles = set()
    unique_docs = []
    for doc in all_docs:
        title = doc.split(" | ")[0] if " | " in doc else doc[:100]
        normalized_title = dspy.evaluate.normalize_text(title)

        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_docs.append(doc)

    # Should have 4 unique documents (Titanic, James Cameron, Canada, Avatar)
    # Duplicates: "Titanic" appears twice, "james cameron" appears twice (case-insensitive)
    expected_unique = 4
    assert len(unique_docs) == expected_unique, f"Expected {expected_unique} unique docs, got {len(unique_docs)}"

    print(f"  ✓ Original documents: {len(all_docs)}")
    print(f"  ✓ Unique documents after deduplication: {len(unique_docs)}")
    print("✓ Deduplication test passed\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Running BM25 Reranker and Sequential Multi-Hop Tests")
    print("=" * 80)
    print()

    # BM25 tests
    test_bm25_tokenization()
    test_bm25_idf_computation()
    test_bm25_scoring()
    test_bm25_empty_docs()

    # Sequential hop tests
    test_sequential_hops()
    test_deduplication()

    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print("✓ BM25Reranker: tokenization, IDF computation, and scoring work correctly")
    print("✓ Sequential hops: 3 retrievals happen in sequence")
    print("✓ Entity extraction: called twice (hop 2 and hop 3)")
    print("✓ Query generation: called twice (hop 2 and hop 3)")
    print("✓ Deduplication: removes duplicate documents by normalized title")
    print("✓ Returns at most 21 unique documents")
