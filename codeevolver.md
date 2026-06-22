## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "3-hop retrieval with entity-extraction-from-passages (ExtractNextQuery), k=25, and interleaved round-robin deduplication"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 3-hop multi-hop retrieval system that retrieves 25 candidates per hop using DSPy's ColBERTv2 retriever. Instead of summarizing passages and gap-filling, hops 2 and 3 use a single shared `ExtractNextQuery` DSPy Signature that reads the raw retrieved passages directly and identifies the most important named entity not yet covered by retrieved titles. This entity-extraction approach preserves specific names mentioned in passages (e.g., actor names, film titles) that would be lost through summarization. After all three hops, candidates are merged via interleaved round-robin deduplication, and the final output is at most 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs three retrieval hops, each with k=25 candidates, using only 2 LM calls total (one for hop 2 and one for hop 3):

- **Hop 1**: Retrieves directly from the raw claim. No LM call is made; the top-10 passage titles are collected for deduplication tracking.

- **Hop 2**: `extract_query` (ExtractNextQuery signature) receives the claim, top-5 raw passages from hop 1, and the semicolon-separated hop-1 titles. It uses ChainOfThought to identify the single most important named entity from the passages that is directly relevant to the claim but not yet retrieved, outputting a short precise search query (typically just the entity name). This entity-first approach preserves specific names mentioned in passages that would be lost through summarization.

- **Hop 3**: `extract_query` is called again with the claim, the combined top-3 passages from hops 1 and 2, and the full semicolon-separated titles from both hops. It identifies the next missing entity and generates a targeted search query for it.

**Merge — interleaved round-robin deduplication**: Rather than naively concatenating hop lists, we iterate round i=0..24 and, for each hop in order (hop1, hop2, hop3), add hop[i] to the final list if its title (the part before " | ") has not been seen before. We stop once 21 unique documents are collected. The final `dspy.Prediction(retrieved_docs=final_docs[:21])` is returned upstream to the evaluator.
