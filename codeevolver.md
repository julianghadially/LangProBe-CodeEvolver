PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: HoVer 3-Hop Fact-Checking Retrieval Pipeline with ColBERTv2 and Resource-Penalty Metric

## ARCHITECTURE SUMMARY:
The system is a multi-hop document retrieval pipeline for the HoVer fact-checking benchmark. Given a claim, it performs three sequential retrieval hops using a remote ColBERTv2 server, with LLM-generated summaries and queries guiding each successive hop. The top-level entry point `HoverMultiHopPipeline` (`hover_pipeline.py`) wraps the core DSPy module `HoverMultiHop` (`hover_program.py`) and injects a thread-safe `CountingRM` (`counting_rm.py`) to track the number of search queries issued.

The metric (`hover_utils.py`) evaluates whether all gold supporting documents appear in the final retrieved set (capped at 21 docs), then applies a soft penalty for search queries beyond a free budget of 2, returning a `ScoreWithFeedback` object that also provides textual feedback for reflective optimization.

## ARCHITECTURE DESCRIPTION:
**What it does:** Given a factual claim, the pipeline retrieves the supporting Wikipedia documents required to verify it, targeting claims that require exactly 3 supporting documents (3-hop reasoning).

**Data flow (`hover_data.py` → `hover_pipeline.py` → `hover_program.py`):**
1. `hoverBench` (in `hover_data.py`) loads the `hover-nlp/hover` HuggingFace dataset, filters to examples requiring up to 3 unique supporting documents, and wraps each as a `dspy.Example` with `claim` as the input field and `supporting_facts` as the label.
2. At runtime, `HoverMultiHopPipeline.forward(claim)` resets the thread-local search counter on `CountingRM`, sets the DSPy context to use `gpt-5.4-nano` (with `reasoning_effort="low"`) and the counting retriever, then delegates to `HoverMultiHop.forward`.
3. `HoverMultiHop` (`hover_program.py`) executes three retrieval hops:
   - **Hop 1:** Retrieves top-7 passages directly from the claim, then summarizes them (`summarize1` ChainOfThought).
   - **Hop 2:** Uses `create_query_hop2` (ChainOfThought on `claim` + `summary_1`) to generate a refined query, retrieves 7 more passages, and produces `summary_2`.
   - **Hop 3:** Uses `create_query_hop3` (ChainOfThought on `claim` + both summaries) to generate a final query and retrieves 7 more passages.
   - Returns a `dspy.Prediction` with all 21 retrieved passages concatenated.
4. After `HoverMultiHop` returns, `HoverMultiHopPipeline` attaches `search_count` (from `CountingRM.get_count()`) to the prediction object.

**Key modules:**
- `hover_pipeline.py`: Top-level pipeline class (`HoverMultiHopPipeline`); configures the LM and `CountingRM`; sets `search_count` on predictions.
- `hover_program.py`: Core DSPy module (`HoverMultiHop`) with 3 retrieval hops, 2 query-generation modules, and 2 summarization modules, all as `dspy.ChainOfThought`.
- `counting_rm.py`: `CountingRM` wraps `dspy.ColBERTv2`; thread-local counters support parallel evaluation; monkey-patches ColBERTv2's HTTP layer to use a pooled session with retry/timeout logic for the remote Modal-hosted server.
- `hover_utils.py`: Defines `discrete_retrieval_eval_with_resource_penalty_and_feedback` — the primary metric. Scores 1.0 for full recall (all gold doc titles found in top-21 retrieved), then subtracts `0.002 × max(0, search_count − 2)`. Returns a `ScoreWithFeedback` with score and a human-readable feedback string naming missing documents.
- `hover_data.py`: `hoverBench` dataset class; filters to 3-hop examples; exposes train/test splits.
- `tracing_setup.py`: Imported as a side-effect in `hover_pipeline.py`; instruments DSPy with OpenTelemetry via `openinference` for span emission across parallel threads.

**Hard constraints enforced by the metric:** The retrieved document list is truncated at 21 entries before evaluation; returning more is silently ignored, never rewarded.
