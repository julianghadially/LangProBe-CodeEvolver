#!/usr/bin/env python3
"""Test script for the new dynamic sequential reasoning architecture."""

import dspy
from langProBe.hover.hover_program import HoverMultiHopPredict

def test_basic_structure():
    """Test that the module initializes correctly."""
    print("Testing module initialization...")
    program = HoverMultiHopPredict()

    # Check that the new planners exist
    assert hasattr(program, 'plan_first_hop'), "Missing plan_first_hop"
    assert hasattr(program, 'plan_next_hop'), "Missing plan_next_hop"
    assert hasattr(program, 'retrieve_k'), "Missing retrieve_k"

    # Check that old modules are removed
    assert not hasattr(program, 'extract_entities'), "Old extract_entities should be removed"
    assert not hasattr(program, 'target_entity'), "Old target_entity should be removed"

    print("✓ Module structure is correct")

def test_helper_methods():
    """Test helper methods."""
    print("\nTesting helper methods...")
    program = HoverMultiHopPredict()

    # Test _extract_title
    doc_with_newline = "Title Here\nContent here..."
    assert program._extract_title(doc_with_newline) == "Title Here"

    doc_without_newline = "Just a single line document"
    assert len(program._extract_title(doc_without_newline)) <= 100

    # Test _summarize_documents
    sample_docs = [
        "Doc 1 Title\nThis is content about Lisa Raymond",
        "Doc 2 Title\nThis talks about Martina Hingis",
        "Doc 3 Title\nFrench Open doubles information"
    ]
    summary = program._summarize_documents(sample_docs)
    assert "Doc 1 Title" in summary
    assert "Doc 2 Title" in summary
    assert "Doc 3 Title" in summary

    # Test empty docs
    empty_summary = program._summarize_documents([])
    assert empty_summary == "No documents retrieved yet"

    print("✓ Helper methods work correctly")

def test_signatures():
    """Test that signatures are defined correctly."""
    print("\nTesting signature definitions...")
    from langProBe.hover.hover_program import FirstHopPlanner, NextHopPlanner

    # Check FirstHopPlanner fields
    first_hop = FirstHopPlanner
    assert 'claim' in first_hop.input_fields
    assert 'reasoning' in first_hop.output_fields
    assert 'search_query' in first_hop.output_fields

    # Check NextHopPlanner fields
    next_hop = NextHopPlanner
    assert 'claim' in next_hop.input_fields
    assert 'previous_queries' in next_hop.input_fields
    assert 'retrieved_titles' in next_hop.input_fields
    assert 'key_facts_found' in next_hop.input_fields
    assert 'information_gap' in next_hop.output_fields
    assert 'reasoning' in next_hop.output_fields
    assert 'search_query' in next_hop.output_fields

    print("✓ Signatures are defined correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Dynamic Sequential Reasoning Implementation")
    print("=" * 60)

    try:
        test_basic_structure()
        test_helper_methods()
        test_signatures()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe implementation is ready. To test with actual retrieval:")
        print("1. Set up a language model: lm = dspy.LM('openai/gpt-4o-mini')")
        print("2. Configure DSPy: dspy.settings.configure(lm=lm)")
        print("3. Set up retriever in pipeline: HoverMultiHopPredictPipeline()")
        print("4. Test with a claim: pipeline(claim='Your test claim here')")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
