```
PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: Three-Hop DSPy Retrieval Pipeline with Query Generation, Summarization, and Resource-Penalized Feedback Metric

## ARCHITECTURE SUMMARY:
HoverMultiHopPipeline (`langProBe/hover/hover_pipeline.py`) is the entry-point wrapper that wires together a DSPy language model (DeepSeek-V4-Flash via GMI Cloud/LiteLLM), a thread-safe ColBERTv2 retrieval model wrapped by CountingRM, and the core HoverMultiHop program (`langProBe/hover/hover_program.py`). HoverMultiHop performs three iterative retrieval hops over a claim: hop 1 retrieves directly on the claim, then summarizes; hops 2 and 3 generate refined queries from prior summaries and retrieve again. The union of passages across all three hops is returned as `retrieved_docs`. The metric in `langProBe/hover/hover_utils.py` checks whether all gold supporting documents appear among the top-21 retrieved, then applies a soft penalty for searches beyond a free budget, returning a ScoreWithFeedback for reflective optimization.

## ARCHITECTURE DESCRIPTION:
The HoverMultiHopPipeline (hover_pipeline.py:28) is the benchmark entry point invoked per claim. Its __init__ configures the LM (DeepSeek-V4-Flash through GMI Cloud via LiteLLM's openai-compatible route with reasoning_effort="high"), wraps a remote ColBERTv2 retriever (hosted on Modal) in CountingRM (counting_rm.py), and instantiates HoverMultiHop (hover_program.py:5). On forward(), it resets the per-thread search counter, runs the program under dspy.context with the configured lm and rm, attaches search_count to the result, and returns it.

HoverMultiHop defines k=7 and four ChainOfThought modules (create_query_hop2, create_query_hop3, summarize1, summarize2) plus a dspy.Retrieve(k=7). Hop 1 retrieves passages from the raw claim and summarizes them. Hop 2 generates a query from claim+summary_1, retrieves, and summarizes with prior context. Hop 3 generates a query from claim+summary_1+summary_2 and retrieves. The final retrieved_docs concatenates all three hops (21 docs max).

CountingRM (counting_rm.py) is a thread-safe wrapper around dspy.ColBERTv2 that uses threading.local counters per evaluation thread, monkey-patches ColBERTv2's HTTP request to use a pooled requests.Session with elevated timeouts (240s) and retries (max 2 with 60s backoff), enabling robust concurrent retrieval.

Dataset loading lives in hover_data.py:9 (hoverBench), which loads the hover-nlp/hover dataset, filters to <=3-hop examples, shuffles deterministically, and produces dspy.Example objects with claim/supporting_facts/label. Tracing is enabled additively by tracing_setup.py (DSPyInstrumentor) to emit OpenTelemetry spans for all DSPy calls.

The optimized metric, discrete_retrieval_eval_with_resource_penalty_and_feedback (hover_utils.py:69), normalizes gold titles from supporting_facts and found titles from the first 21 retrieved_docs, computes success=1.0 if gold⊆found else 0.0, calculates a penalty of 0.002 per search beyond a free budget of 2, returns max(0, success−penalty), and packages this with textual feedback (missing docs, search count, penalty) as a dspy ScoreWithFeedback for GEPA-style reflective optimization. There is no hard cap on searches; the only hard constraint is returning ≤21 documents.
```
