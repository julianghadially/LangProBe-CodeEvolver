"""Enable DSPy -> OpenTelemetry auto-instrumentation for the Hover pipeline.

Importing this module patches DSPy so every Predict / ChainOfThought / ReAct /
Retrieve / LM call emits an OTel span on the global TracerProvider that the
CodeEvolver orchestrator installs. Additive only -- it attaches to whatever
provider already exists and never creates or replaces one.

The instrumentor also propagates OTel context across DSPy's parallel worker
threads (dspy.Evaluate / dspy.Parallel), so spans stay correctly nested when
the pipeline runs with num_threads > 1.
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
