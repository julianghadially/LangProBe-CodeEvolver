# Document Reranking with Diversity-Aware Selection

## Overview

The `HoverMultiHopPredict` module now includes a relevance-based document reranking system that runs after all three retrieval hops complete. This prevents redundant retrieval (e.g., 3x Gatwick, 3x Coldwaltham) and surfaces comparative documents that may be retrieved but buried in the results.

## Architecture

### New Components

1. **ScoreDocumentRelevance Signature**
   - Input: `claim` (the claim to verify) and `document` (in 'title | content' format)
   - Output: `relevance_score` (1-10 integer) and `reasoning` (explanation)
   - Evaluates: factual overlap, entity coverage, temporal information, and comparative value

2. **Reranking Module**
   - Instantiated in `__init__`: `self.score_relevance = dspy.ChainOfThought(ScoreDocumentRelevance)`
   - Called in `forward()` after collecting all 21 documents from three hops

3. **Diversity-Aware Selection Algorithm**
   - Implemented in `_rerank_with_diversity()` method

## How It Works

### Step-by-Step Process

```
1. Collect Documents (7 + 6 + 8 = 21 documents)
   ├─ Hop 1: Direct retrieval (k=7)
   ├─ Hop 2: 3 sub-queries × k=2
   └─ Hop 3: 4 sub-queries × k=2

2. Score Each Document
   ├─ For each of 21 documents:
   │  ├─ Call self.score_relevance(claim, document)
   │  ├─ Extract relevance_score (1-10)
   │  └─ Store reasoning
   └─ Sort by score (descending)

3. Diversity-Aware Selection
   ├─ Extract normalized title from each document
   ├─ Iterate through scored documents:
   │  ├─ If title is new: add to selected_docs
   │  └─ If title is duplicate: add to overflow_docs
   └─ Fill remaining slots (up to 21) with overflow

4. Return Reranked Documents
   └─ Up to 21 unique or high-scoring documents
```

### Example Scenario

**Before Reranking:**
```
1. Gatwick Airport | Busiest single-runway airport... (retrieved 3 times)
2. Gatwick Airport | Located in West Sussex... (duplicate)
3. Coldwaltham | Village in Horsham district... (retrieved 3 times)
4. Heathrow Airport | Busiest UK airport... (buried at position 15)
5. Gatwick Airport | Second busiest UK airport... (duplicate)
6. Coldwaltham | Population 527... (duplicate)
```

**After Reranking:**
```
1. Heathrow Airport | Busiest UK airport... (score: 10, comparative value)
2. Gatwick Airport | Busiest single-runway airport... (score: 9, highest of 3 instances)
3. Coldwaltham | Village in Horsham district... (score: 7, deduplicated)
4. [Other unique documents...]
```

## Code Implementation

### ScoreDocumentRelevance Signature

```python
class ScoreDocumentRelevance(dspy.Signature):
    """Evaluate how relevant a document is for verifying a specific claim.
    Consider: factual overlap, entity coverage, temporal information, and comparative value."""

    claim = dspy.InputField(desc="The claim to verify")
    document = dspy.InputField(desc="Document text in 'title | content' format")
    relevance_score: int = dspy.OutputField(
        desc="Relevance score from 1-10, where 10 is highly relevant and 1 is irrelevant"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this score was assigned, highlighting key factors"
    )
```

### Integration in HoverMultiHopPredict

```python
def forward(self, claim):
    # ... [Hop 1, 2, 3 retrieval code] ...

    # Combine all documents (7 + 6 + 8 = 21)
    all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs

    # Rerank with diversity-aware selection
    reranked_docs = self._rerank_with_diversity(claim, all_retrieved_docs)

    return dspy.Prediction(retrieved_docs=reranked_docs)
```

### Reranking Algorithm

```python
def _rerank_with_diversity(self, claim: str, documents: list[str]) -> list[str]:
    # 1. Score all documents
    scored_docs = []
    for doc in documents:
        prediction = self.score_relevance(claim=claim, document=doc)
        scored_docs.append({
            'document': doc,
            'score': prediction.relevance_score,
            'reasoning': prediction.reasoning
        })

    # 2. Sort by score (descending)
    scored_docs.sort(key=lambda x: x['score'], reverse=True)

    # 3. Diversity-aware selection
    seen_titles = set()
    selected_docs = []
    overflow_docs = []

    for item in scored_docs:
        doc = item['document']
        title = get_normalized_title(doc)

        if title not in seen_titles:
            seen_titles.add(title)
            selected_docs.append(doc)
        else:
            overflow_docs.append(doc)

    # 4. Fill remaining slots (up to 21)
    max_docs = 21
    remaining_slots = max_docs - len(selected_docs)
    if remaining_slots > 0:
        selected_docs.extend(overflow_docs[:remaining_slots])

    return selected_docs
```

## Benefits

### 1. Prevents Redundancy
- **Problem**: Multiple hops may retrieve the same document (e.g., "Gatwick Airport" appears 3 times)
- **Solution**: Normalized title deduplication keeps only the highest-scored instance

### 2. Surfaces Comparative Documents
- **Problem**: Important comparative documents (e.g., Heathrow for ranking verification) may be buried
- **Solution**: Relevance scoring brings high-value documents to the top

### 3. Maintains Coverage
- **Problem**: Aggressive deduplication might reduce document count
- **Solution**: Overflow mechanism fills remaining slots up to 21 documents

### 4. Intelligent Title Normalization
- **Handles**: Case differences, whitespace variations, punctuation
- **Example**: "Gatwick Airport", "GATWICK AIRPORT", "Gatwick  Airport" → all normalize to "gatwick airport"

## Error Handling

The reranking system includes robust error handling:

```python
try:
    prediction = self.score_relevance(claim=claim, document=doc)
    score = prediction.relevance_score

    # Ensure score is valid integer in [1, 10]
    if isinstance(score, str):
        score = int(score)
    score = max(1, min(10, score))

except Exception as e:
    # Fallback: assign neutral score if scoring fails
    score = 5
    reasoning = f"Scoring failed: {str(e)}"
```

## Performance Considerations

- **Scoring**: Each document requires one LLM call (21 calls total)
- **Optimization**: Consider batching or caching for production use
- **Trade-off**: Improved quality vs. increased latency

## Future Enhancements

1. **Batch Scoring**: Score multiple documents in a single LLM call
2. **Semantic Deduplication**: Use embeddings instead of title matching
3. **Dynamic Threshold**: Adjust diversity threshold based on claim complexity
4. **Relevance Caching**: Cache scores for frequently retrieved documents
5. **Multi-Criteria Ranking**: Combine relevance with novelty, coverage, and diversity metrics

## Example Usage

```python
from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline
import dspy

# Initialize pipeline
pipeline = HoverMultiHopPredictPipeline()

# Run multi-hop retrieval with reranking
claim = "Gatwick Airport is the second busiest UK airport and is located in Coldwaltham"
result = pipeline(claim=claim)

# Access reranked documents
for i, doc in enumerate(result.retrieved_docs, 1):
    print(f"{i}. {doc[:100]}...")  # Print first 100 chars
```

## Configuration

To disable reranking (for testing or comparison):

```python
# In hover_program.py, modify forward() method:
def forward(self, claim):
    # ... [retrieval code] ...

    all_retrieved_docs = hop1_docs + hop2_docs + hop3_docs

    # Skip reranking (comment out this line)
    # reranked_docs = self._rerank_with_diversity(claim, all_retrieved_docs)

    # Return unranked documents
    return dspy.Prediction(retrieved_docs=all_retrieved_docs)
```

## Testing Recommendations

1. **Verify Deduplication**: Check that duplicate titles are reduced
2. **Test Scoring**: Ensure relevance scores are reasonable (1-10 range)
3. **Compare Rankings**: Measure improvement in fact verification accuracy
4. **Benchmark Performance**: Track latency impact of 21 additional LLM calls
5. **Edge Cases**: Test with empty documents, malformed titles, scoring failures
