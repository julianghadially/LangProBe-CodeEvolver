```
PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval_with_resource_penalty_and_feedback

## ARCHITECTURE TITLE: 4-Hop Retrieval with Fruitless-Query Feedback, Hop-4 Slot Reservation, and Stronger Entity-Targeting Prompt

## ARCHITECTURE SUMMARY:
`HoverMultiHopPipeline` (`langProBe/hover/hover_pipeline.py`) is the top-level DSPy module evaluated per row. It wraps `HoverMultiHop` (`langProBe/hover/hover_program.py`), a fixed 4-hop retrieval chain, inside a `CountingRM` (`langProBe/hover/counting_rm.py`) instrumented retrieval model that tracks how many ColBERT searches are fired per example.

The inner `HoverMultiHop` module performs four sequential hops with k=7 per retrieval. Hop 1 retrieves on the raw claim. Hops 2–4 each call `ChainOfThought(IdentifyNextTarget)` to identify the next uncovered Wikipedia entity. Three key improvements over prior iterations: (1) **Incremental deduplication** — `seen_titles` is updated after each hop so each hop only sees genuinely new documents; (2) **Fruitless-query feedback** — any hop that returns 0 new unique documents has its query added to `fruitless_queries`, which is passed to subsequent LM calls via the new `fruitless_queries` field in `IdentifyNextTarget`, steering the LM away from redundant searches; (3) **Slot reservation** — hops 1–3 are capped at 14 slots in the final output, guaranteeing hop 4 contributes up to 7 documents and fixing the prior "slot starvation" bug.

The `IdentifyNextTarget` prompt has been strengthened with CRITICAL nuances for person-centric entities (e.g., "fronted by [person]" requires that person's own article) and exact Wikipedia article title targeting for seasons/episodes/songs.

## ARCHITECTURE DESCRIPTION:
**What the program does**: Given a factual claim from the HoVer multi-hop fact-checking dataset, the pipeline retrieves the set of Wikipedia articles that support or refute the claim. The dataset (`hover_data.py:hoverBench`) loads `hover-nlp/hover` from HuggingFace, filters to examples requiring exactly 2–3 supporting documents, and wraps each as a `dspy.Example(claim=..., supporting_facts=..., label=...)` with `claim` as the input key.

**Key modules and responsibilities**:
- `hover_pipeline.py / HoverMultiHopPipeline`: Outer pipeline; instantiates the LM (`openai/gpt-5.4-nano`, reasoning_effort="low"), wraps `dspy.ColBERTv2` in `CountingRM`, resets the per-thread search counter before each forward pass, then reads the count back into `result.search_count` after the inner program finishes. Inherits `LangProBeDSPyMetaProgram`.
- `hover_program.py / HoverMultiHop`: Core 4-hop DSPy module with k=7 documents per retrieval. Hop 1 retrieves directly on the claim. Hops 2, 3, and 4 each use `ChainOfThought(IdentifyNextTarget)` to identify the single most important uncovered entity. Deduplication is incremental (per-hop, not post-hoc): `get_new_unique()` updates `seen_titles` after each hop and flags queries returning 0 new docs as fruitless. Slot allocation caps hops 1–3 at 14 docs and appends all of hop 4's new docs, ensuring hop 4 is never starved. Final result capped at 21 documents.
- `hover_program.py / IdentifyNextTarget`: DSPy Signature with three input fields (`claim`, `retrieved_passages`, `fruitless_queries`) that instructs the LM to enumerate named entities in the claim (with nuances for person-centric references and exact season/episode article titles), check which are covered by dedicated article titles in retrieved passages, and output a single Wikipedia article title or entity name. Step 5 in the prompt explicitly forbids repeating any query from `fruitless_queries`.
- `counting_rm.py / CountingRM`: Thread-safe retrieval wrapper using `threading.local()` for per-thread counters (safe for `dspy.Evaluate` parallel mode). Also monkey-patches `dspy.dsp.colbertv2` to use a shared, connection-pooled `requests.Session` with configurable timeout (60 s default) and retry backoff, addressing DNS resolution failures under concurrent load.
- `tracing_setup.py`: On import, calls `DSPyInstrumentor().instrument()` (openinference) once, attaching OpenTelemetry spans to every DSPy Predict/Retrieve/LM call for CodeEvolver's IterationArchitect trace inspection.
- `hover_utils.py`: Metric logic. `discrete_retrieval_eval_with_resource_penalty_and_feedback` normalizes gold and found title sets, checks subset containment (binary success), applies `PENALTY_PER_SEARCH * max(0, search_count - 2)` soft penalty, and returns `ScoreWithFeedback(score, feedback)` where feedback details missing documents, counts, and penalty values.

**Data flow**: `claim` → `HoverMultiHopPipeline.forward` → reset counter → `HoverMultiHop.forward` (hop1: ColBERT on raw claim k=7, dedup; hop2: `IdentifyNextTarget`(fruitless_queries) → entity → ColBERT k=7, dedup+fruitless-flag; hop3: same with accumulated fruitless list; hop4: same; slot-reserve hops1-3 to 14, append hop4_new; cap at 21) → `dspy.Prediction(retrieved_docs=[≤21 passages])` → attach `search_count` → metric compares `retrieved_docs[:21]` titles against `supporting_facts[*].key` → `ScoreWithFeedback`.

**Metric being optimized**: A composite float in [0, 1] equal to `1.0 - penalty` if all gold documents are found (0.0 otherwise), where penalty = 0.002 × max(0, searches − 2). The hard output constraint is ≤ 21 returned documents; there is no hard cap on search count, only the soft penalty.
```
