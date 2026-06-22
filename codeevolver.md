## PARENT_MODULE_PATH: "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"

## ARCHITECTURE TITLE: "3-hop retrieval with GapFillingQuery signature, explicit retrieved_titles input, k=21, and interleaved round-robin deduplication"

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline wraps HoverMultiHop, a 3-hop multi-hop retrieval system that retrieves 21 candidates per hop using DSPy's ColBERTv2 retriever. Hop 3 now uses a dedicated `GapFillingQuery` DSPy Signature whose instruction directs the LM to compare claim entities against already-retrieved document titles and target the missing entity. The `retrieved_titles` input (a semicolon-separated list of the top-6 titles from hops 1+2) makes the coverage gap explicit, replacing the old `summary_1` input. After all three hops, candidates are merged via interleaved round-robin deduplication, and summarization modules receive only top-7 passages to keep summaries crisp. The final output is at most 21 unique documents.

## ARCHITECTURE DESCRIPTION:
HoverMultiHopPipeline is the top-level entry point. It configures the language model (gpt-5.4-nano with low reasoning effort) and a CountingRM-wrapped ColBERTv2 retriever, then delegates all retrieval logic to HoverMultiHop via a dspy.context call.

HoverMultiHop performs three retrieval hops, each with k=21 candidates:

- **Hop 1**: Retrieves directly from the raw claim. The top-7 passages are passed to `summarize1` (claim, passages → summary) to build an initial summary without overloading the LM with all 21 docs.

- **Hop 2**: `create_query_hop2` (claim, summary_1 → query) generates a targeted follow-up query. The top-7 of the resulting 21 passages go to `summarize2` (claim, context, passages → summary) to extend the context.

- **Hop 3**: `create_query_hop3` uses the new `GapFillingQuery` signature (claim, retrieved_titles, summary_2 → query). Before calling it, `retrieved_titles` is constructed as a semicolon-separated list of the normalised titles from the top-6 docs of hops 1 and 2. The signature's docstring instructs the LM to compare these explicit titles against the entities in the claim and generate a query targeting the missing one. `summary_1` is no longer passed to hop 3 (its context is already captured in `summary_2`).

**Merge — interleaved round-robin deduplication**: Rather than naively concatenating hop lists (which lets hop1 dominate), we iterate round i=0..20 and, for each hop in order (hop1, hop2, hop3), add hop[i] to the final list if its title (the part before " | ") has not been seen before. We stop once 21 unique documents are collected. This guarantees balanced contribution from all three hops while eliminating duplicate Wikipedia articles. The final `dspy.Prediction(retrieved_docs=final_docs[:21])` is returned upstream to the evaluator.
