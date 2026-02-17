# HoverMultiHop Entity-First: Quick Start Guide

## 🎯 What Was Done

Refactored `langProBe/hover/hover_program.py` with an **entity-extraction-first approach** that:
- ✅ Extracts entities, relationships, and key facts BEFORE retrieval
- ✅ Creates targeted queries for entity clusters
- ✅ Uses k=30/20/15 for three hops (primary/secondary/relationship entities)
- ✅ Applies relevance-based reranking to select final 21 documents
- ✅ Maximizes entity coverage within constraints

## 📦 New Components

### 1. ClaimEntityExtractor (DSPy Signature)
```python
# Extracts structured entity information from claims
extraction = entity_extractor(claim="Your claim")
# Returns: primary_entities, secondary_entities, relationships, key_facts
```

### 2. EntityBasedQueryGenerator (DSPy Signature)
```python
# Generates targeted queries for entity clusters
query = query_generator(
    claim=claim,
    entity_cluster=entities,
    relationships=relationships,
    previous_findings=context
)
```

### 3. DocumentRelevanceScorer (DSPy Signature)
```python
# Scores documents for entity coverage (0-10)
scoring = doc_scorer(
    claim=claim,
    document=doc,
    target_entities=entities,
    target_relationships=relationships
)
```

### 4. Enhanced HoverMultiHop Module
```python
# Main class with entity-first retrieval pipeline
hover = HoverMultiHop()
result = hover(claim="Your claim")
# Returns: 21 documents with maximum entity coverage
```

## 🚀 How It Works

```
Claim → Extract Entities → Multi-Hop Retrieval → Rerank → Top 21 Docs
         ↓                  ↓                      ↓
         • Primary         Hop 1: k=30           Score all 65
         • Secondary       Hop 2: k=20           Select best 21
         • Relationships   Hop 3: k=15           Max coverage
         • Key facts
```

## 💻 Quick Usage

### Option 1: Use the Pipeline (Recommended)
```python
import dspy
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline

# Configure (once)
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", api_key="YOUR_KEY"),
    rm=dspy.ColBERTv2(url="YOUR_COLBERT_URL")
)

# Use
pipeline = HoverMultiHopPipeline()
result = pipeline(claim="Your claim here")
print(f"Retrieved {len(result.retrieved_docs)} documents")
```

### Option 2: Use HoverMultiHop Directly
```python
from langProBe.hover.hover_program import HoverMultiHop

hover = HoverMultiHop()
hover.setup_lm("openai/gpt-4o-mini", api_key="YOUR_KEY")

with dspy.context(rm=your_retriever):
    result = hover(claim="Your claim")
```

### Option 3: Use Entity Extractor Standalone
```python
from langProBe.hover.hover_program import ClaimEntityExtractor

extractor = dspy.ChainOfThought(ClaimEntityExtractor)
extraction = extractor(claim="Your claim")

print(extraction.primary_entities)
print(extraction.secondary_entities)
print(extraction.relationships)
print(extraction.key_facts)
```

## 📊 What Changed

| Before | After |
|--------|-------|
| Generic queries | Entity-targeted queries |
| k=7 fixed for all hops | k=30/20/15 adaptive |
| 21 docs (7×3) | 65 docs → rerank to 21 |
| No entity tracking | Explicit entity coverage |
| No quality filtering | Relevance-based scoring |

## 🔍 Example Walkthrough

**Claim**: "The director of The Matrix also directed Cloud Atlas with Tom Hanks."

### Step 1: Entity Extraction
```
Primary: The Matrix, Cloud Atlas, Tom Hanks
Secondary: Wachowski directors, film production
Relationships: Same director for both films, Tom Hanks in Cloud Atlas
Key Facts: Director names, release dates, cast information
```

### Step 2: Targeted Retrieval
```
Hop 1 (k=30): "The Matrix Cloud Atlas Tom Hanks directors"
              → 30 docs about these main subjects

Hop 2 (k=20): "Wachowski brothers filmography"
              → 20 docs about bridging entities

Hop 3 (k=15): "Wachowski directed Matrix Cloud Atlas Tom Hanks"
              → 15 docs verifying relationships
```

### Step 3: Reranking
```
65 total docs → Score each for entity coverage → Select top 21
```

## 📁 Important Files

### Modified
- `langProBe/hover/hover_program.py` - **Main implementation**

### Documentation (Created)
- `IMPLEMENTATION_README.md` - Comprehensive guide
- `HOVER_REFACTORING_SUMMARY.md` - Technical details
- `ARCHITECTURE_COMPARISON.md` - Visual comparisons
- `example_usage.py` - Working examples
- `QUICK_START.md` - This file

## ✅ Testing Checklist

```python
# Test with these example claims:
claims = [
    "Marie Curie won Nobel Prizes in both Physics and Chemistry.",
    "The Eiffel Tower was built for the 1889 World's Fair in Paris.",
    "Barack Obama was born in Hawaii and became the 44th President.",
]

for claim in claims:
    result = pipeline(claim=claim)
    assert len(result.retrieved_docs) <= 21
    print(f"✓ {claim[:50]}... -> {len(result.retrieved_docs)} docs")
```

## 🎓 Key Benefits

1. **Better Entity Coverage**: Explicitly tracks and maximizes entity coverage
2. **Higher Quality**: Reranking selects best documents from larger pool
3. **Targeted Retrieval**: Entity-specific queries are more focused
4. **Adaptive Strategy**: Different k values for different entity types
5. **Transparent**: Clear separation of extraction, retrieval, and ranking phases

## 🔧 Configuration Tips

### Adjust Retrieval Sizes
```python
hover = HoverMultiHop()
hover.retrieve_hop1 = dspy.Retrieve(k=40)  # More primary entities
hover.retrieve_hop2 = dspy.Retrieve(k=15)  # Fewer secondary
```

### Change Language Model
```python
# Faster, cheaper
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"))

# More capable
dspy.configure(lm=dspy.LM("openai/gpt-4"))
```

## 🐛 Common Issues

### "No LM is loaded"
**Solution**: Call `dspy.configure(lm=...)` before using any modules

### Returns < 21 docs
**Normal**: Happens if retriever doesn't return enough unique documents

### Slow performance
**Consider**:
- Use faster LM (gpt-3.5-turbo vs gpt-4)
- Implement parallel document scoring
- Cache entity extractions

## 📈 Next Steps

1. **Test**: Run with your actual claims and retriever
2. **Benchmark**: Compare accuracy vs original implementation
3. **Optimize**: Profile and optimize slow parts if needed
4. **Extend**: Add entity-type-specific handling if desired

## 🔗 Learn More

- Full guide: `IMPLEMENTATION_README.md`
- Technical details: `HOVER_REFACTORING_SUMMARY.md`
- Architecture: `ARCHITECTURE_COMPARISON.md`
- Examples: Run `python example_usage.py`

---

**Status**: ✅ Implementation complete and ready to use
**Compatibility**: ✅ Backward compatible with existing code
**Testing**: ⏳ Ready for integration testing

## 📞 Quick Help

**Q: How do I know it's working?**
```python
result = pipeline(claim="Test claim")
print(f"Got {len(result.retrieved_docs)} docs (should be ≤21)")
print(f"First doc: {result.retrieved_docs[0][:100]}")
```

**Q: How do I see what entities were extracted?**
```python
# Add this in hover_program.py forward() method to debug:
print(f"Primary: {primary_entities}")
print(f"Secondary: {secondary_entities}")
```

**Q: Can I use my own retriever?**
```python
# Yes! Use with dspy.context
with dspy.context(rm=my_custom_retriever):
    result = hover(claim=claim)
```

---

Happy fact-checking! 🎉
