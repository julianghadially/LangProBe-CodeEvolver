## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "3-hop retrieval with k=21 per hop, interleaved round-robin deduplication, and top-7 passage summarization"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 3-hop multi-hop retrieval system that retrieves 21 candidates per hop (up from 7) using DSPy's ColBERTv2 retriever. Each hop generates a progressively refined query using ChainOfThought modules (create_query_hop2, create_query_hop3) conditioned on prior summaries. After all three hops, candidate documents are merged via interleaved round-robin deduplication: position i is drawn from hop1, hop2, then hop3 in turn, skipping titles already seen. Summarization modules (summarize1, summarize2) receive only the top-7 passages from each hop to keep summaries crisp. The final output is at most 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs three retrieval hops, each with k=21 candidates:

- **Hop 1**: Retrieves directly from the raw claim. The top-7 passages are passed to `summarize1` (claim, passages → summary) to build an initial summary without overloading the LM with all 21 docs.

- **Hop 2**: `create_query_hop2` (claim, summary_1 → query) generates a targeted follow-up query. The top-7 of the resulting 21 passages go to `summarize2` (claim, context, passages → summary) to extend the context.

- **Hop 3**: `create_query_hop3` (claim, summary_1, summary_2 → query) generates a gap-filling query using both prior summaries. Its 21 passages feed directly into the merge step.

**Merge — interleaved round-robin deduplication**: Rather than naively concatenating hop lists (which lets hop1 dominate), we iterate round i=0..20 and, for each hop in order (hop1, hop2, hop3), add hop[i] to the final list if its title (the part before " | ") has not been seen before. We stop once 21 unique documents are collected. This guarantees balanced contribution from all three hops while eliminating duplicate Wikipedia articles. The final `dspy.Prediction(retrieved_docs=final_docs[:21])` is returned upstream to the evaluator.
