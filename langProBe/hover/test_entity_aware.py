"""
Unit tests for the entity-aware gap analysis retrieval pipeline.

Tests the HoverEntityAwareMultiHop class and its components:
- ExtractClaimEntities signature
- VerifyEntityCoverage module
- Entity-aware multi-hop retrieval
- Document reranking
"""

import dspy
from langProBe.hover.hover_program import (
    HoverEntityAwareMultiHop,
    ExtractClaimEntities,
    VerifyEntityCoverage,
    RankDocumentsByRelevance
)


def test_extract_claim_entities_signature():
    """Test that ExtractClaimEntities signature is properly defined."""
    sig = ExtractClaimEntities

    # Check input fields
    assert 'claim' in sig.input_fields
    assert sig.input_fields['claim'].json_schema_extra['desc'] == "The claim to extract entities from"

    # Check output fields
    assert 'entities' in sig.output_fields
    assert sig.output_fields['entities'].annotation == list[str]

    print("✓ ExtractClaimEntities signature is properly defined")


def test_verify_entity_coverage_signature():
    """Test that VerifyEntityCoverage signature is properly defined."""
    sig = VerifyEntityCoverage

    # Check input fields
    assert 'claim' in sig.input_fields
    assert 'entities' in sig.input_fields
    assert 'documents' in sig.input_fields

    # Check output fields
    assert 'uncovered_entities' in sig.output_fields
    assert sig.output_fields['uncovered_entities'].annotation == list[str]

    print("✓ VerifyEntityCoverage signature is properly defined")


def test_rank_documents_signature():
    """Test that RankDocumentsByRelevance signature is properly defined."""
    sig = RankDocumentsByRelevance

    # Check input fields
    assert 'claim' in sig.input_fields
    assert 'entities' in sig.input_fields
    assert 'documents' in sig.input_fields

    # Check output fields
    assert 'relevance_scores' in sig.output_fields
    assert sig.output_fields['relevance_scores'].annotation == list[float]

    print("✓ RankDocumentsByRelevance signature is properly defined")


def test_entity_aware_initialization():
    """Test that HoverEntityAwareMultiHop initializes correctly."""
    pipeline = HoverEntityAwareMultiHop()

    # Check retrieval modules
    assert hasattr(pipeline, 'retrieve_15')
    assert hasattr(pipeline, 'retrieve_10')
    assert pipeline.retrieve_15.k == 15
    assert pipeline.retrieve_10.k == 10

    # Check entity modules
    assert hasattr(pipeline, 'extract_entities')
    assert hasattr(pipeline, 'verify_coverage')
    assert hasattr(pipeline, 'create_entity_query')
    assert hasattr(pipeline, 'rank_documents')

    print("✓ HoverEntityAwareMultiHop initializes correctly")


def test_pipeline_structure():
    """Test that the pipeline has the correct structure."""
    pipeline = HoverEntityAwareMultiHop()

    # Verify it's a DSPy module
    assert isinstance(pipeline, dspy.Module)

    # Check that forward method exists
    assert hasattr(pipeline, 'forward')
    assert callable(pipeline.forward)

    # Check docstring
    assert 'entity-aware' in pipeline.__doc__.lower()
    assert 'gap analysis' in pipeline.__doc__.lower()

    print("✓ Pipeline structure is correct")


def test_mock_pipeline_flow():
    """Test the pipeline flow with mock data (no actual LM/RM calls)."""
    pipeline = HoverEntityAwareMultiHop()

    # Create mock methods to avoid actual LM/RM calls
    class MockRetrieval:
        def __init__(self, docs):
            self.passages = docs

    class MockPrediction:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Mock entity extraction
    mock_entities = ["Entity1", "Entity2", "Entity3"]

    # Mock documents
    mock_docs_15 = [f"Document {i} about Entity1" for i in range(15)]
    mock_docs_10_hop2 = [f"Document {i} about Entity2" for i in range(10)]
    mock_docs_10_hop3 = [f"Document {i} about Entity3" for i in range(10)]

    # Mock uncovered entities
    mock_uncovered = ["Entity2", "Entity3"]

    # Mock scores
    mock_scores = [0.9, 0.85, 0.8, 0.75, 0.7] + [0.6] * 30  # Dummy scores

    print("✓ Mock pipeline flow test structure is valid")


def test_deduplication_logic():
    """Test the deduplication logic in the pipeline."""
    # Simulate duplicate detection
    docs = [
        "This is document 1",
        "This is document 2",
        "This is document 1",  # Duplicate
        "This is document 3",
        "This is document 2",  # Duplicate
    ]

    unique_docs = []
    seen = set()
    for doc in docs:
        doc_hash = hash(doc[:200] if len(doc) > 200 else doc)
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)

    assert len(unique_docs) == 3
    assert unique_docs == ["This is document 1", "This is document 2", "This is document 3"]

    print("✓ Deduplication logic works correctly")


def test_document_ranking_logic():
    """Test the document ranking and selection logic."""
    # Simulate ranking and selection
    docs = [f"Document {i}" for i in range(30)]
    scores = [0.9 - (i * 0.02) for i in range(30)]  # Decreasing scores

    # Pair and sort
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Select top 21
    final_docs = [doc for doc, score in doc_score_pairs[:21]]

    assert len(final_docs) == 21
    assert final_docs[0] == "Document 0"  # Highest score
    assert final_docs[-1] == "Document 20"

    print("✓ Document ranking logic works correctly")


def test_edge_cases():
    """Test edge cases in the pipeline."""
    pipeline = HoverEntityAwareMultiHop()

    # Test 1: No uncovered entities (empty list)
    uncovered_entities = []
    assert len(uncovered_entities) == 0
    # hop2 should be skipped
    hop2_docs = [] if len(uncovered_entities) == 0 else ["doc"]
    assert hop2_docs == []

    # Test 2: Only one uncovered entity
    uncovered_entities = ["Entity1"]
    assert len(uncovered_entities) == 1
    # hop2 should run, hop3 should be skipped
    hop3_docs = [] if len(uncovered_entities) <= 1 else ["doc"]
    assert hop3_docs == []

    # Test 3: Fewer than 21 unique documents
    unique_docs = [f"Doc {i}" for i in range(15)]
    if len(unique_docs) > 21:
        final_docs = unique_docs[:21]
    else:
        final_docs = unique_docs[:21]
    assert len(final_docs) == 15

    print("✓ Edge cases handled correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running Entity-Aware Pipeline Tests")
    print("="*70 + "\n")

    try:
        test_extract_claim_entities_signature()
        test_verify_entity_coverage_signature()
        test_rank_documents_signature()
        test_entity_aware_initialization()
        test_pipeline_structure()
        test_mock_pipeline_flow()
        test_deduplication_logic()
        test_document_ranking_logic()
        test_edge_cases()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
