# Quick Start Guide: Document Reranking

## What Was Added?

A relevance-based document reranking system that automatically:
1. ✅ **Scores** each of the 21 retrieved documents (1-10 relevance scale)
2. ✅ **Deduplicates** documents with the same normalized title
3. ✅ **Surfaces** high-value comparative documents (e.g., Heathrow for ranking claims)
4. ✅ **Maintains** coverage by backfilling with overflow documents

## How to Use It

### Basic Usage (Reranking is Automatic)

```python
from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline

# Initialize pipeline
pipeline = HoverMultiHopPredictPipeline()

# Run retrieval - reranking happens automatically!
claim = "Gatwick Airport is the second busiest UK airport"
result = pipeline(claim=claim)

# Access reranked documents
print(f"Retrieved {len(result.retrieved_docs)} documents")
for i, doc in enumerate(result.retrieved_docs[:5], 1):
    print(f"{i}. {doc[:80]}...")
```

### Expected Output

**Before Reranking:**
- 21 documents (may include 3x Gatwick, 3x Coldwaltham)
- Comparative documents buried in results

**After Reranking:**
- 15-21 unique/high-scoring documents
- Deduplicated (only best instance of each title)
- Ranked by relevance (high-value documents first)

## How It Works

```
1. Multi-hop retrieval collects 21 documents
   └─ Hop 1 (k=7) + Hop 2 (3×k=2) + Hop 3 (4×k=2) = 21 docs

2. Score each document (ScoreDocumentRelevance)
   └─ LLM evaluates: factual overlap, entity coverage, comparative value
   └─ Output: relevance_score (1-10) + reasoning

3. Diversity-aware selection
   └─ Sort by score (descending)
   └─ Keep highest-scored instance of each unique title
   └─ Fill remaining slots with overflow documents

4. Return reranked documents
   └─ Up to 21 unique/high-scoring documents
```

## Testing

Run the test suite to verify everything works:

```bash
python langProBe/hover/test_reranking.py
```

**Expected Output:**
```
======================================================================
HOVER MULTI-HOP RERANKING TEST SUITE
======================================================================

Testing title normalization:
  ✓ 'Gatwick Airport | Content here...' → 'gatwick airport'
  ✓ 'GATWICK AIRPORT | Content here...' → 'gatwick airport'
  ...

ALL TESTS PASSED ✓
```

## Configuration

### To Disable Reranking

Edit `langProBe/hover/hover_program.py`:

```python
def forward(self, claim):
    # ... [retrieval code] ...
    all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs

    # Comment out these lines to disable reranking
    # reranked_docs = self._rerank_with_diversity(claim, all_retrieved_docs)
    # return dspy.Prediction(retrieved_docs=reranked_docs)

    # Return unranked documents
    return dspy.Prediction(retrieved_docs=all_retrieved_docs)
```

## Documentation

📄 **RERANKING_EXAMPLE.md** - Comprehensive guide with examples
📊 **RERANKING_FLOW.txt** - Visual flow diagram and algorithm details
📋 **IMPLEMENTATION_SUMMARY.md** - Complete change summary
🧪 **test_reranking.py** - Test suite with demonstrations

## Key Benefits

| Feature | Before | After |
|---------|--------|-------|
| **Duplicates** | 3x Gatwick, 3x Coldwaltham | Deduplicated (1x each) |
| **Comparative Docs** | Heathrow at position 15 | Heathrow at position 1 |
| **Ordering** | Retrieval order | Relevance-ranked |
| **Coverage** | 21 docs (with duplicates) | 15-21 unique/scored docs |

## Performance

- **Additional Cost**: 21 LLM scoring calls per claim
- **Time Complexity**: O(n log n) where n = 21
- **Optimization**: Consider batch scoring or parallel execution for production

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure you're in the correct directory
```bash
cd /workspace
python langProBe/hover/test_reranking.py
```

### Issue: DSPy context error
**Solution**: Reranking requires DSPy LM configuration
```python
import dspy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4"))
```

### Issue: Scoring fails with error
**Solution**: Fallback mechanism automatically assigns neutral score (5)
- Check error reasoning in document metadata
- Verify LLM configuration is correct

## Example: Real-World Usage

```python
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline

# Configure DSPy with your LLM
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4"))

# Initialize pipeline
pipeline = HoverMultiHopPredictPipeline()

# Test claims
claims = [
    "Gatwick Airport is the second busiest UK airport and is located in Coldwaltham",
    "Heathrow Airport is the busiest airport in the United Kingdom",
    "Horsham is a market town in West Sussex, England"
]

# Run retrieval with reranking
for claim in claims:
    print(f"\nClaim: {claim}")
    result = pipeline(claim=claim)
    print(f"Retrieved {len(result.retrieved_docs)} documents:")

    # Show top 3 reranked documents
    for i, doc in enumerate(result.retrieved_docs[:3], 1):
        title = doc.split(' | ')[0] if ' | ' in doc else doc[:50]
        print(f"  {i}. {title}")
```

**Sample Output:**
```
Claim: Gatwick Airport is the second busiest UK airport...
Retrieved 18 documents:
  1. Heathrow Airport
  2. Gatwick Airport
  3. Coldwaltham

Claim: Heathrow Airport is the busiest airport...
Retrieved 16 documents:
  1. Heathrow Airport
  2. Gatwick Airport
  3. London

...
```

## Next Steps

1. ✅ Run test suite: `python langProBe/hover/test_reranking.py`
2. 📚 Read full documentation: `RERANKING_EXAMPLE.md`
3. 🔍 Review flow diagram: `RERANKING_FLOW.txt`
4. 🚀 Try it with real claims in your pipeline!

## Questions?

- See `IMPLEMENTATION_SUMMARY.md` for technical details
- Check `RERANKING_EXAMPLE.md` for comprehensive examples
- Review `test_reranking.py` for code examples

---

**TL;DR**: Document reranking is now automatic! It scores, deduplicates, and ranks your retrieved documents by relevance. Just use the pipeline as normal - reranking happens behind the scenes. 🎉
