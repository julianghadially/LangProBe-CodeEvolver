"""Enable DSPy -> OpenTelemetry auto-instrumentation for the RAGQAArenaTech pipeline.

Importing this module patches DSPy so every Predict / ChainOfThought / Module / LM
call emits an OTel span on the global TracerProvider that the CodeEvolver orchestrator
installs. Additive only -- it attaches to whatever provider already exists and never
creates or replaces one.

The instrumentor also propagates OTel context across DSPy's parallel worker threads
(dspy.Evaluate / dspy.Parallel), so spans stay correctly nested when the pipeline runs
with num_threads > 1.

Mirrors langProBe/hover/tracing_setup.py rather than importing it: `import
langProBe.hover.tracing_setup` would execute langProBe/hover/__init__.py, building the
whole hover benchmark registry (HF `datasets`, HoverMultiHop()) inside the RAGQA eval
process for no benefit.

Scope note: this covers the LM and dspy.Module spans only. Unlike hover -- which
retrieves through dspy.Retrieve/dspy.ColBERTv2, both patched by OpenInference --
RAGQA retrieves through a plain HTTP client that auto-instrumentation cannot see, so
that span is emitted manually by @traceable("retriever") in RAGQAArenaTech_retrieval.py.
"""
from openinference.instrumentation.dspy import DSPyInstrumentor

_INSTRUMENTED = False


def setup_dspy_tracing() -> None:
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    DSPyInstrumentor().instrument()
    _INSTRUMENTED = True


setup_dspy_tracing()
