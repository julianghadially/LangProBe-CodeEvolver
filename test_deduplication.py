#!/usr/bin/env python3
"""
Test script to verify the deduplication and reranking logic works correctly.
"""


def test_deduplicate_and_rerank():
    """Test the deduplication and reranking logic with mock data."""

    # Mock document data to simulate retrieval results
    hop1_docs = [
        "Doc A | Content about topic A",
        "Doc B | Content about topic B",
        "Doc C | Content about topic C",
        "Doc D | Content about topic D",
        "Doc E | Content about topic E",
        "Doc F | Content about topic F",
        "Doc G | Content about topic G",
        "Doc H | Content about topic H",
        "Doc I | Content about topic I",
        "Doc J | Content about topic J",
        "Doc K | Content about topic K",
        "Doc L | Content about topic L",
        "Doc M | Content about topic M",
        "Doc N | Content about topic N",
        "Doc O | Content about topic O",
    ]

    hop2_docs = [
        "Doc A | Content about topic A",  # Duplicate from hop1
        "Doc P | Content about topic P",
        "Doc B | Content about topic B",  # Duplicate from hop1
        "Doc Q | Content about topic Q",
        "Doc R | Content about topic R",
        "Doc S | Content about topic S",
        "Doc T | Content about topic T",
        "Doc U | Content about topic U",
        "Doc V | Content about topic V",
        "Doc W | Content about topic W",
        "Doc X | Content about topic X",
        "Doc Y | Content about topic Y",
        "Doc Z | Content about topic Z",
        "Doc AA | Content about topic AA",
        "Doc AB | Content about topic AB",
    ]

    hop3_docs = [
        "Doc A | Content about topic A",  # Duplicate from hop1 and hop2
        "Doc AC | Content about topic AC",
        "Doc B | Content about topic B",  # Duplicate from hop1 and hop2
        "Doc AD | Content about topic AD",
        "Doc C | Content about topic C",  # Duplicate from hop1
        "Doc AE | Content about topic AE",
        "Doc AF | Content about topic AF",
        "Doc AG | Content about topic AG",
        "Doc AH | Content about topic AH",
        "Doc AI | Content about topic AI",
        "Doc AJ | Content about topic AJ",
        "Doc AK | Content about topic AK",
        "Doc AL | Content about topic AL",
        "Doc AM | Content about topic AM",
        "Doc AN | Content about topic AN",
    ]

    # Inline implementation of the deduplication logic for testing
    def _deduplicate_and_rerank(hop1_docs, hop2_docs, hop3_docs, top_k=21):
        doc_tracker = {}

        for hop_idx, hop_docs in enumerate([hop1_docs, hop2_docs, hop3_docs], start=1):
            for position, doc in enumerate(hop_docs):
                title = doc.split(" | ")[0] if " | " in doc else doc

                if title not in doc_tracker:
                    doc_tracker[title] = {
                        'document': doc,
                        'hop_appearances': [],
                        'positions': []
                    }

                doc_tracker[title]['hop_appearances'].append(hop_idx)
                doc_tracker[title]['positions'].append((hop_idx, position))

        scored_docs = []
        for title, info in doc_tracker.items():
            doc = info['document']
            hop_appearances = info['hop_appearances']
            positions = info['positions']

            cross_hop_score = len(set(hop_appearances))

            position_scores = []
            for hop_idx, position in positions:
                hop_weight = 1.0 - (hop_idx - 1) * 0.05
                position_score = hop_weight / (position + 1)
                position_scores.append(position_score)

            avg_position_score = sum(position_scores) / len(position_scores)
            relevance_score = (0.6 * cross_hop_score) + (0.4 * avg_position_score * 10)

            scored_docs.append({
                'document': doc,
                'title': title,
                'score': relevance_score,
                'cross_hop_count': cross_hop_score,
                'avg_position_score': avg_position_score
            })

        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return [item['document'] for item in scored_docs[:top_k]]

    # Run deduplication
    result = _deduplicate_and_rerank(hop1_docs, hop2_docs, hop3_docs, top_k=21)

    # Verify results
    print("=" * 80)
    print("DEDUPLICATION AND RERANKING TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal documents retrieved: {len(hop1_docs) + len(hop2_docs) + len(hop3_docs)}")
    print(f"Unique documents after deduplication: {len(set([d.split(' | ')[0] for d in hop1_docs + hop2_docs + hop3_docs]))}")
    print(f"Final returned documents: {len(result)}")
    print(f"\nExpected: 21 documents")
    print(f"Actual: {len(result)} documents")
    print(f"Test PASSED: {len(result) == 21}")

    # Check for duplicates in result
    titles = [doc.split(" | ")[0] for doc in result]
    unique_titles = set(titles)
    print(f"\nNo duplicates in result: {len(titles) == len(unique_titles)}")

    # Show top 10 results
    print("\nTop 10 documents (should prioritize cross-hop appearances):")
    for i, doc in enumerate(result[:10], 1):
        title = doc.split(" | ")[0]
        print(f"{i:2d}. {title}")

    # Verify that Doc A and Doc B (appeared in all 3 hops) are ranked highly
    top_5_titles = [doc.split(" | ")[0] for doc in result[:5]]
    print(f"\nDoc A (appears in all 3 hops) in top 5: {'Doc A' in top_5_titles}")
    print(f"Doc B (appears in all 3 hops) in top 5: {'Doc B' in top_5_titles}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_deduplicate_and_rerank()
