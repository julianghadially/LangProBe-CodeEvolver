```
PARENT_MODULE_PATH: langProBe.RAGQAArenaTech.RAGQAArenaTech_pipeline.RAGQAArenaTechPipeline
METRIC_MODULE_PATH: langProBe.RAGQAArenaTech.metric.ragqa_semantic_f1_feedback

## ARCHITECTURE TITLE: Multi-hop DSPy RAG (SimplifiedBaleen) over a warm HTTP retriever, scored by SemanticF1
## ARCHITECTURE SUMMARY:
RAGQAArenaTech is a multi-hop retrieval-augmented QA benchmark over the LoTTE "technology" corpus. A thin pipeline wrapper (`RAGQAArenaTech_pipeline.RAGQAArenaTechPipeline`) owns LM configuration and an injected retriever client, delegating all reasoning to an inner DSPy module — `SimplifiedBaleen` in `RAGQAArenaTech_program.py`. Retrieval is served by an external long-running process exposing `/api/search`; the in-process `HTTPEmbeddingRetriever` client (`RAGQAArenaTech_retrieval.py`) is a thin, connection-pooled, deepcopy-stable singleton over that server. The optimized metric (`metric.ragqa_semantic_f1_feedback`) wraps `dspy.evaluate.SemanticF1` with the GEPA `ScoreWithFeedback` contract, adding judge reasoning plus directional recall/precision hints for CodeEvolver reflection.

## ARCHITECTURE DESCRIPTION:
The eval entry point is `RAGQAArenaTechPipeline` (`RAGQAArenaTech_pipeline.py`), a `dspy.Module` subclass under `LangProBeDSPyMetaProgram`. Its `forward(question)` runs the inner program under `dspy.context(lm=self.lm)`, where `self.lm` is DeepSeek-V4-Flash via DeepInfra/LiteLLM (key from `DEEPINFRA_API_KEY`, never threaded through DSPy so it stays out of OTel traces). DSPy disk/memory caching is disabled at import so each eval exercises the real LM and retrieval.

The inner program is `SimplifiedBaleen` (`RAGQAArenaTech_program.py`): two hops of `dspy.ChainOfThought(GenerateSearchQuery)` (`RAGQAArenaTech_utils.py` defines the signature: context + question → query), each followed by a `self.search(query, k=5)` call, with passages deduplicated via `langProBe.dspy_program.deduplicate`. After `max_hops` it synthesizes a final `response` via `dspy.ChainOfThought("context, question -> response")`. The program holds no corpus/index and performs no IO — it just calls the injected retriever — so the optimizer can freely swap hop count, query generation, and synthesis.

The retriever is `HTTPEmbeddingRetriever` (`RAGQAArenaTech_retrieval.py`): a stateless handle over a process-wide pooled `requests.Session` (pool_maxsize=32) talking to a local warm server (default `http://localhost:8894/api/search`, configurable via `RAGQA_RETRIEVER_URL`). It returns `topk[].text` truncated to 4000 chars, with retry/backoff on connection errors. `__deepcopy__`/`__copy__` return `self` so the optimizer never clones retrieval infrastructure. `get_default_retriever()` lazily builds a double-checked-locked singleton.

The metric `ragqa_semantic_f1_feedback` (`metric.py`) constructs a `dspy.ChainOfThought(SemanticRecallPrecision)` judge on `openai/gpt-5.4-mini`, scores the prediction's `response` (falling back to `answer`) against the gold `example.response`, and returns `ScoreWithFeedback(score=f1, feedback=...)`. The feedback string embeds F1/recall/precision, the question, gold and system responses, the judge's `reasoning`, and a directional hint (recall-low → retrieve/cover more; precision-low → be more faithful; zero → reconsider both retrieval and generation). A companion `ragqa_semantic_f1` returns the bare float (bool ≥ 0.66 under tracing).
```
