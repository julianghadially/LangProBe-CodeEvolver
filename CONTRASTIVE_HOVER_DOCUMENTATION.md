# Negative Feedback Retrieval Architecture with Explicit Query Contrast Learning

## Overview

This implementation enhances the `HoverMultiHop` system with a sophisticated contrastive learning approach for document retrieval. The architecture explicitly learns what **NOT** to retrieve by generating negative queries alongside positive queries, creating contrast between wanted and unwanted information.

## Architecture Components

### 1. ContrastiveQuerySignature Classes

Three signature classes define the dual-query generation interface:

#### `ContrastiveQuerySignature` (Base)
- **Purpose**: General-purpose signature for contrastive query generation
- **Inputs**:
  - `claim`: The claim to verify
  - `previous_summary`: Summary of previously retrieved documents
  - `retrieved_passages`: Recently retrieved passages to analyze
  - `cumulative_negative_context`: Accumulated negative context from previous hops
- **Outputs**:
  - `positive_query`: Targets missing information gaps
  - `negative_query`: Represents irrelevant content patterns to avoid

#### `ContrastiveQuerySignatureHop2`
- **Purpose**: Specialized for Hop 2 query generation
- **Inputs**:
  - `claim`: The claim to verify
  - `summary_1`: Summary from Hop 1
- **Outputs**: Same dual-query structure

#### `ContrastiveQuerySignatureHop3`
- **Purpose**: Specialized for Hop 3 with cumulative negative context
- **Inputs**:
  - `claim`: The claim to verify
  - `summary_1`: Summary from Hop 1
  - `summary_2`: Summary from Hop 2
  - `cumulative_negative_context`: All negative queries from previous hops
- **Outputs**: Same dual-query structure

### 2. ContrastiveQueryGenerator Modules

Implemented using DSPy's `ChainOfThought` with the signature classes:

```python
self.create_query_hop2 = dspy.ChainOfThought(ContrastiveQuerySignatureHop2)
self.create_query_hop3 = dspy.ChainOfThought(ContrastiveQuerySignatureHop3)
```

These modules replace the original simple query generators and produce both positive and negative queries for each hop.

### 3. Custom Reranking Layer

The `rerank_with_contrast()` method implements the core contrastive scoring mechanism:

#### Algorithm:
1. **Retrieve**: Fetch k=15 documents using the positive query
2. **Score**: Compute contrast score for each document:
   ```
   contrast_score = α × positive_similarity + β × negative_dissimilarity
   ```
   - `positive_similarity`: Proportion of positive query terms in document
   - `negative_dissimilarity`: 1 - (proportion of negative query terms in document)
   - Default weights: α=0.6, β=0.4
3. **Rerank**: Sort documents by contrast score (descending)
4. **Select**: Return top k=7 documents

#### Key Features:
- **Explicit Contrast**: Documents are scored on both what they contain (positive) and what they don't contain (negative)
- **Weighted Scoring**: Configurable balance between positive attraction and negative repulsion
- **Automatic Deduplication**: Naturally avoids similar irrelevant documents through negative scoring

### 4. Cumulative Negative Context Tracking

The system maintains a history of all negative queries across hops:

```python
self.negative_queries_history = []
```

- **Hop 2**: Generates first negative query based on Hop 1 results
- **Hop 3**: Receives cumulative context from Hops 1 & 2, generating a more informed negative query
- **Purpose**: Build progressive understanding of what to avoid, preventing repeated retrieval of similar irrelevant content

## Retrieval Flow

### Hop 1: Baseline Retrieval
```
claim → retrieve(k=15) → top 7 documents → summarize
```
- No contrastive learning yet (no prior context)
- Establishes initial information baseline

### Hop 2: First Contrastive Hop
```
claim + summary_1 → generate_positive_query + generate_negative_query
                  → retrieve(k=15, positive_query)
                  → rerank_with_contrast(positive_query, negative_query)
                  → top 7 documents → summarize
```
- First use of contrastive reranking
- Negative query based on irrelevant patterns from Hop 1

### Hop 3: Cumulative Contrastive Hop
```
claim + summary_1 + summary_2 + cumulative_negative_context
    → generate_positive_query + generate_negative_query
    → retrieve(k=15, positive_query)
    → rerank_with_contrast(positive_query, negative_query)
    → top 7 documents
```
- Uses accumulated negative context from all previous hops
- Most informed contrastive reranking

### Total Documents: 7 + 7 + 7 = 21 (maximum)

## Implementation Details

### Class: `HoverMultiHop`

#### Constructor Parameters:
- `alpha` (float, default=0.6): Weight for positive query similarity
- `beta` (float, default=0.4): Weight for negative query dissimilarity

#### Key Attributes:
- `k_retrieve = 15`: Number of documents to retrieve per hop
- `k_final = 7`: Number of documents to keep after reranking
- `negative_queries_history`: List tracking all negative queries

#### Key Methods:

##### `compute_contrast_score(doc_text, positive_query, negative_query)`
Computes weighted contrast score for a document.

**Returns**: Float score where higher values indicate better matches

**Formula**:
```
pos_score = |positive_terms ∩ doc_terms| / |positive_terms|
neg_score = 1 - (|negative_terms ∩ doc_terms| / |negative_terms|)
contrast_score = α × pos_score + β × neg_score
```

##### `rerank_with_contrast(documents, positive_query, negative_query)`
Reranks k=15 documents using contrast scoring, returns top k=7.

**Args**:
- `documents`: List of 15 document strings
- `positive_query`: Query for relevant information
- `negative_query`: Query for irrelevant patterns

**Returns**: List of 7 top-ranked documents

##### `forward(claim)`
Main retrieval pipeline with contrastive learning.

**Returns**: `dspy.Prediction` containing:
- `retrieved_docs`: All 21 documents (7 per hop)
- `negative_queries`: List of negative queries from all hops
- `positive_queries`: List of positive queries from Hops 2 & 3

## Benefits of This Architecture

### 1. Explicit Negative Feedback
- System learns what NOT to retrieve, not just what to retrieve
- Prevents cycles of retrieving similar irrelevant documents

### 2. Progressive Refinement
- Each hop builds on accumulated knowledge of irrelevant patterns
- Cumulative negative context becomes more sophisticated over time

### 3. Balanced Scoring
- Configurable α/β weights allow tuning attraction vs. repulsion
- Can adapt to different retrieval scenarios

### 4. Scalable Reranking
- Retrieves broader set (k=15) for better coverage
- Intelligently narrows to high-quality subset (k=7)
- Total documents remain at optimal 21

### 5. Interpretable Queries
- Negative queries provide explicit insight into what system considers irrelevant
- Useful for debugging and understanding retrieval behavior

## Example Usage

```python
import dspy
from langProBe.hover.hover_program import HoverMultiHop

# Configure DSPy with your LM and retriever
dspy.configure(
    lm=dspy.LM("openai/gpt-4"),
    rm=dspy.ColBERTv2(url="http://your-retriever-url")
)

# Initialize with custom weights (optional)
model = HoverMultiHop(alpha=0.7, beta=0.3)

# Run retrieval
claim = "The Eiffel Tower was completed in 1889 for the World's Fair"
result = model(claim=claim)

# Access results
print(f"Retrieved {len(result.retrieved_docs)} documents")
print(f"Negative queries used: {result.negative_queries}")
print(f"Positive queries used: {result.positive_queries}")
```

## Advanced: Customizing Contrast Scoring

The default `compute_contrast_score()` uses simple term overlap. For production use, consider:

### 1. Embedding-Based Scoring
```python
def compute_contrast_score(self, doc_text, positive_query, negative_query):
    # Use sentence transformers or similar
    doc_emb = self.encoder.encode(doc_text)
    pos_emb = self.encoder.encode(positive_query)
    neg_emb = self.encoder.encode(negative_query)

    pos_sim = cosine_similarity(doc_emb, pos_emb)
    neg_sim = cosine_similarity(doc_emb, neg_emb)

    return self.alpha * pos_sim + self.beta * (1 - neg_sim)
```

### 2. BM25 Scoring
```python
from rank_bm25 import BM25Okapi

def compute_contrast_score(self, doc_text, positive_query, negative_query):
    tokenized_docs = [doc_text.split()]

    bm25_pos = BM25Okapi(tokenized_docs)
    pos_score = bm25_pos.get_scores(positive_query.split())[0]

    bm25_neg = BM25Okapi(tokenized_docs)
    neg_score = bm25_neg.get_scores(negative_query.split())[0]

    # Normalize and combine
    pos_norm = pos_score / (pos_score + neg_score + 1e-6)
    neg_norm = neg_score / (pos_score + neg_score + 1e-6)

    return self.alpha * pos_norm + self.beta * (1 - neg_norm)
```

### 3. Cross-Encoder Reranking
```python
from sentence_transformers import CrossEncoder

def compute_contrast_score(self, doc_text, positive_query, negative_query):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    pos_score = cross_encoder.predict([(positive_query, doc_text)])[0]
    neg_score = cross_encoder.predict([(negative_query, doc_text)])[0]

    return self.alpha * pos_score + self.beta * (1 - neg_score)
```

## Performance Considerations

### Computational Complexity
- **Per Hop**: O(k_retrieve × scoring_complexity)
- **Total**: 3 hops × 15 documents × scoring = 45 scoring operations
- **Scoring**: O(|doc| × |query|) for term overlap (can be optimized with embeddings)

### Memory Usage
- Stores k=15 documents per hop temporarily
- Final storage: 21 documents
- Negative queries history: ~3 strings (lightweight)

### Optimization Tips
1. **Batch Scoring**: Score all documents in parallel if using neural models
2. **Cache Embeddings**: Reuse document embeddings across hops if possible
3. **Early Stopping**: Skip scoring documents with very low positive similarity
4. **Adaptive k**: Dynamically adjust k_retrieve based on query difficulty

## Evaluation Metrics

To assess the effectiveness of contrastive learning:

### 1. Relevance Improvement
- Compare document relevance before and after reranking
- Measure precision@k for k=7 vs. first 7 of original 15

### 2. Diversity Increase
- Measure document diversity in final set vs. non-contrastive baseline
- Check for reduction in near-duplicate retrievals

### 3. Negative Query Quality
- Analyze how well negative queries capture irrelevant patterns
- Measure overlap between negative queries and actually irrelevant documents

### 4. Cumulative Effect
- Track improvement from Hop 2 to Hop 3 with cumulative negative context
- Measure information gain per hop

## Troubleshooting

### Issue: Negative queries too similar to positive queries
**Solution**: Enhance signature instructions to emphasize distinction, or add post-processing to filter overlapping terms

### Issue: Reranking not improving results
**Solution**: Adjust α/β weights, or switch to embedding-based scoring for better semantic understanding

### Issue: Cumulative negative context becoming too large
**Solution**: Implement summarization of negative queries or keep only most recent N queries

### Issue: Hop 1 retrieval quality poor (no contrast yet)
**Solution**: Consider initializing with domain-specific negative patterns, or expand Hop 1 to k=10 then rerank to 7

## Future Extensions

1. **Learned Weights**: Train α/β weights on validation data
2. **Dynamic k**: Adjust k_retrieve and k_final based on query complexity
3. **Multi-Granularity Negatives**: Generate negative queries at different levels (term, sentence, topic)
4. **Reinforcement Learning**: Use RL to optimize negative query generation
5. **Cross-Hop Deduplication**: Explicitly check for document similarity across hops
6. **Negative Query Templates**: Provide few-shot examples to improve negative query quality

## Testing

Run the test suite to verify implementation:

```bash
python test_contrastive_hover.py
```

Expected output: All tests pass, confirming:
- ✓ Signature definitions
- ✓ Initialization with correct parameters
- ✓ Contrast scoring favors relevant documents
- ✓ Reranking produces 7 documents from 15
- ✓ Architecture summary

## References

This implementation is inspired by research in:
- Contrastive learning for information retrieval
- Multi-hop question answering
- Negative feedback in neural search systems
- Query reformulation with explicit constraints

## License

This code is part of the langProBe project. See project LICENSE for details.
