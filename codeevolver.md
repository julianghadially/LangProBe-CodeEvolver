```
PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: DSPy 3-Hop ColBERT Retrieval Pipeline with Resource-Penalized Evaluation

## ARCHITECTURE SUMMARY:
`HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) is the top-level DSPy module evaluated per row. It wraps `HoverMultiHop` (`langProBe/hover/hover_program.py`), a fixed 3-hop retrieval chain, inside a `CountingRM` (`langProBe/hover/counting_rm.py`) instrumented retrieval model that tracks how many ColBERT searches are fired per example.

The inner `HoverMultiHop` module performs three sequential hops: an initial retrieve on the raw claim, a second retrieve on a query derived from the claim plus a first-hop summary, and a third retrieve on a query derived from both summaries. Each hop calls `dspy.Retrieve(k=7)` against a remote ColBERT server and a `dspy.ChainOfThought` LM call to either summarize passages or generate the next query. All 21 retrieved passages (7 × 3 hops) are returned as `pred.retrieved_docs`.

The metric (`hover_utils.py:discrete_retrieval_eval_with_resource_penalty_and_feedback`) checks whether all gold supporting-document titles appear in the top-21 retrieved documents, then applies a soft search-count penalty (0.002 per query beyond a free budget of 2), and returns a `ScoreWithFeedback` object carrying both the composite float score and a natural-language feedback string for reflective optimization.

## ARCHITECTURE DESCRIPTION:
**What the program does**: Given a factual claim from the HoVer multi-hop fact-checking dataset, the pipeline retrieves the set of Wikipedia articles that support or refute the claim. The dataset (`hover_data.py:hoverBench`) loads `hover-nlp/hover` from HuggingFace, filters to examples requiring exactly 2–3 supporting documents, and wraps each as a `dspy.Example(claim=..., supporting_facts=..., label=...)` with `claim` as the input key.

**Key modules and responsibilities**:
- `hover_pipeline.py / HoverMultiHopPipeline`: Outer pipeline; instantiates the LM (`openai/gpt-5.4-nano`, reasoning_effort="low"), wraps `dspy.ColBERTv2` in `CountingRM`, resets the per-thread search counter before each forward pass, then reads the count back into `result.search_count` after the inner program finishes. Inherits `LangProBeDSPyMetaProgram`.
- `hover_program.py / HoverMultiHop`: Core 3-hop DSPy module. Hop 1 retrieves directly on the claim. Hop 2 generates a query via `ChainOfThought("claim,summary_1->query")` and retrieves. Hop 3 generates a query via `ChainOfThought("claim,summary_1,summary_2->query")` and retrieves. Summarization between hops uses `ChainOfThought("claim,passages->summary")` and `ChainOfThought("claim,context,passages->summary")`. Each `dspy.Retrieve(k=7)` call returns 7 passages.
- `counting_rm.py / CountingRM`: Thread-safe retrieval wrapper using `threading.local()` for per-thread counters (safe for `dspy.Evaluate` parallel mode). Also monkey-patches `dspy.dsp.colbertv2` to use a shared, connection-pooled `requests.Session` with configurable timeout (60 s default) and retry backoff, addressing DNS resolution failures under concurrent load.
- `tracing_setup.py`: On import, calls `DSPyInstrumentor().instrument()` (openinference) once, attaching OpenTelemetry spans to every DSPy Predict/Retrieve/LM call for CodeEvolver's IterationArchitect trace inspection.
- `hover_utils.py`: Metric logic. `discrete_retrieval_eval_with_resource_penalty_and_feedback` normalizes gold and found title sets, checks subset containment (binary success), applies `PENALTY_PER_SEARCH * max(0, search_count - 2)` soft penalty, and returns `ScoreWithFeedback(score, feedback)` where feedback details missing documents, counts, and penalty values.

**Data flow**: `claim` → `HoverMultiHopPipeline.forward` → reset counter → `HoverMultiHop.forward` (3 × ColBERT retrieval + 4 × ChainOfThought LM calls) → `dspy.Prediction(retrieved_docs=[21 passages])` → attach `search_count` → metric compares `retrieved_docs[:21]` titles against `supporting_facts[*].key` → `ScoreWithFeedback`.

**Metric being optimized**: A composite float in [0, 1] equal to `1.0 - penalty` if all gold documents are found (0.0 otherwise), where penalty = 0.002 × max(0, searches − 2). The hard output constraint is ≤ 21 returned documents; there is no hard cap on search count, only the soft penalty.
```
