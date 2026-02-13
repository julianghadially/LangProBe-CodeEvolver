# Quick Start Guide: HotPot with Serper + Firecrawl

## Prerequisites

1. **Set Environment Variables:**
   ```bash
   export SERPER_KEY="your_serper_api_key"
   export FIRECRAWL_KEY="your_firecrawl_api_key"
   ```

2. **Configure DSPy Language Model:**
   ```python
   import dspy

   # Example with OpenAI
   lm = dspy.OpenAI(model="gpt-4", max_tokens=1000)
   dspy.settings.configure(lm=lm)
   ```

## Basic Usage

```python
from langProPlus.hotpotGEPA.hotpot_program import HotpotMultiHopPredict

# Initialize the program
program = HotpotMultiHopPredict()

# Ask a multi-hop question
question = "What is the capital of the country where the Eiffel Tower is located?"
result = program(question=question)

print(f"Answer: {result.answer}")
```

## Using the Pipeline

```python
from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPredictPipeline

# Initialize the pipeline
pipeline = HotpotMultiHopPredictPipeline()

# Run inference
result = pipeline(question="Your question here")
print(result.answer)
```

## How It Works

### Step-by-Step Flow

1. **Query Generation (Hop 1)**
   - Input: Original question
   - Output: Optimized search query
   - Example: "capital country Eiffel Tower" → "Eiffel Tower location Paris France"

2. **Search & Scrape (Hop 1)**
   - Serper searches: `"Eiffel Tower location Paris France site:wikipedia.org"`
   - Returns top 5 Wikipedia URLs
   - Firecrawl scrapes the #1 result
   - Returns full page markdown (up to 15,000 chars)

3. **Summarization (Hop 1)**
   - Input: Question + Full Wikipedia page content
   - Output: Concise summary focusing on relevant facts
   - Example: "The Eiffel Tower is in Paris, France. France is a country in Europe."

4. **Query Generation (Hop 2)**
   - Input: Original question + Summary from Hop 1
   - Output: Search query for second hop
   - Example: "What is the capital of France"

5. **Search & Scrape (Hop 2)**
   - Serper searches: `"capital of France site:wikipedia.org"`
   - Firecrawl scrapes the top result
   - Returns full page about France/Paris

6. **Summarization (Hop 2)**
   - Input: Question + Previous summary + New Wikipedia page
   - Output: Summary connecting both hops
   - Example: "Paris is the capital of France. This is where the Eiffel Tower is located."

7. **Answer Generation**
   - Input: Question + Both summaries
   - Output: Minimal factoid answer
   - Example: "Paris"

## Example Questions

### Simple Two-Hop
```python
question = "What year was the director of Titanic born?"
# Expected: 1954 (James Cameron)
```

### Complex Multi-Hop
```python
question = "What is the population of the city where the headquarters of the company that makes the iPhone is located?"
# Expected: ~160,000 (Cupertino, California - Apple headquarters)
```

### Historical Two-Hop
```python
question = "What university did the author of 'The Origin of Species' attend?"
# Expected: University of Edinburgh / Christ's College, Cambridge (Charles Darwin)
```

## Troubleshooting

### No Results Found
**Issue:** "No search results found for: [query]"

**Solutions:**
- Check that `SERPER_KEY` is valid and has credits
- Verify the query is not too specific
- Try broadening the search terms

### Scraping Failed
**Issue:** Falls back to snippet only

**Solutions:**
- Check that `FIRECRAWL_KEY` is valid and has credits
- Some Wikipedia pages may have scraping protection
- The snippet fallback should still provide some context

### Wrong Answers
**Issue:** Answer doesn't match expected result

**Solutions:**
- Check if the summaries contain the right information (add logging)
- Verify Wikipedia pages being scraped are relevant
- Consider increasing `max_length` from 15,000 to get more context
- Adjust the LM temperature (lower = more deterministic)

### Token Limits Exceeded
**Issue:** LM token limit error

**Solutions:**
- Reduce `max_length` in `_search_and_scrape()` (currently 15,000)
- Use a model with larger context window
- Improve summarization to be more concise

## Performance Tips

### Speed Optimization
- Use `dspy.OpenAI(model="gpt-3.5-turbo")` for faster inference
- Reduce `max_length` to 10,000 if full pages not needed
- Cache scraped pages (implement caching layer)

### Cost Optimization
- Serper: ~$5 per 1,000 searches
- Firecrawl: Varies by plan (~$20-50/month typical)
- LM costs: Depends on model (GPT-4 vs GPT-3.5)
- **Total for 100 questions:** ~$1-5 depending on model

### Accuracy Optimization
- Use more capable LM (e.g., GPT-4, Claude-3)
- Increase `max_length` to 20,000+ for complex questions
- Add more search results (increase `num_results` in `_search_and_scrape`)
- Experiment with different summarization prompts

## API Rate Limits

### Serper
- Free tier: 2,500 searches/month
- Paid: 10,000+ searches/month
- Rate limit: ~100 requests/second

### Firecrawl
- Free tier: 500 scrapes/month
- Paid: 10,000+ scrapes/month
- Rate limit: Varies by plan

### Recommendations
- For development: Free tiers sufficient
- For evaluation: Consider paid plans
- For production: Use caching to reduce API calls

## Debugging

Add logging to see intermediate results:

```python
import logging
logging.basicConfig(level=logging.INFO)

# In _search_and_scrape, add:
print(f"Query: {query}")
print(f"Search results: {len(results)}")
print(f"Top URL: {top_url}")
print(f"Scraped content length: {len(scraped.markdown)}")
```

## Next Steps

1. **Test with sample questions** (see examples above)
2. **Run evaluation suite** to compare with baseline
3. **Tune parameters** (max_length, num_results)
4. **Monitor API costs** and optimize as needed
5. **Implement caching** for repeated queries

## Support

For issues or questions:
- Check `HOTPOT_SERPER_FIRECRAWL_MIGRATION.md` for detailed documentation
- Review `CHANGES_SUMMARY.md` for what changed
- Run `test_hotpot_serper_firecrawl.py` to verify setup
