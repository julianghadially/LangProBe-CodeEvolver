"""HTTP client for the RAGQAArenaTech dense retriever.

The corpus + 3.7GB embedding index no longer live in this process. They are served
by a separate, long-running local server (``ragqa-retriever-server``) that loads the
index **once at boot** and stays warm across every eval run. This module is a thin
client over its ``/api/search`` endpoint.

Why this exists: loading the 3.7GB ``index.pt`` inside the eval process was re-paid on
every fresh eval run (and was even triggered at import time via the benchmark
registry), causing eval startup to intermittently time out. Talking to a warm local
server makes retrieval a cheap localhost call and fully decouples the heavy IO from the
optimizable pipeline code -- the optimizer can never touch or reload the index.

Configure the endpoint with ``RAGQA_RETRIEVER_URL`` (default
``http://localhost:8894/api/search``). Start the server before running any RAGQA eval;
see the ``ragqa-retriever-server`` repo.

Thread-safety: ``HTTPEmbeddingRetriever`` holds no mutable per-call state and shares a
connection-pooled ``requests.Session``, so concurrent calls from dspy ``Evaluate``'s
thread pool are safe.
"""

import os
import sys
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from ._tracing import traceable

try:  # OTel is optional -- this module must stay importable outside CodeEvolver.
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover
    _otel_trace = None

DEFAULT_URL = "http://localhost:8894/api/search"

# Shared, connection-pooled HTTP session for all searches. A pooled Session resolves
# the host once and reuses kept-alive connections across threads, avoiding per-search
# DNS/connection churn under concurrent load. ``pool_maxsize`` must be >= the eval
# thread count (currently up to 16) so concurrent threads each reuse a connection.
_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


def _record_doc_scores(topk: list[dict]) -> None:
    """Stamp per-document relevance scores onto the active ``search`` span.

    ``search`` returns passage text only, so the server's ranking signal is otherwise
    discarded before anything can observe it. Uses OpenInference's own attribute name
    (``retrieval.documents.{i}.document.score``) so these spans carry the same score
    signal as hover's auto-instrumented dspy.ColBERTv2 spans. Document text is left
    out -- it is already on the span as ``ce.output``.

    Never raises: tracing must not be able to break retrieval, and the module must stay
    usable with no OpenTelemetry installed.
    """
    if _otel_trace is None:
        return
    try:
        span = _otel_trace.get_current_span()
        if span is None or not span.is_recording():
            return
        for i, doc in enumerate(topk):
            span.set_attribute(
                f"retrieval.documents.{i}.document.score", float(doc.get("score") or 0.0)
            )
    except Exception:
        pass


class HTTPEmbeddingRetriever:
    """Thin client over the dense-retriever server's ``/api/search`` endpoint.

    Preserves the original in-process ``EmbeddingRetriever`` interface --
    ``search(query, k) -> list[str]`` -- so the program/pipeline are unchanged. The
    server returns full passages; this client owns the ``max_characters`` truncation
    to keep behavior identical to the original retriever.
    """

    def __init__(
        self,
        url: str | None = None,
        max_characters: int = 4000,
        timeout: int = 60,
        max_retries: int = 2,
        retry_backoff: int = 5,
    ):
        self.url = url or os.environ.get("RAGQA_RETRIEVER_URL", DEFAULT_URL)
        self.max_characters = max_characters
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    # One span per logical retrieval (the decorator wraps the whole retry loop, not each
    # HTTP attempt). Records the hop's query text, k, the returned passages, and -- on
    # failure -- the error, which is what OpenInference gives hover for free via
    # dspy.Retrieve/dspy.ColBERTv2 but cannot give a plain HTTP client.
    @traceable("retriever")
    def search(self, query: str, k: int = 5) -> list[str]:
        for attempt in range(self.max_retries + 1):
            try:
                res = _SESSION.get(
                    self.url, params={"query": query, "k": k}, timeout=self.timeout
                )
                res.raise_for_status()
                res_json = res.json()
                if "topk" not in res_json:
                    raise ValueError(
                        f"Retriever server returned an unexpected response: {res_json}"
                    )
                topk = res_json["topk"][:k]
                # Capture ranking scores before the projection below discards them.
                _record_doc_scores(topk)
                return [doc["text"][: self.max_characters] for doc in topk]
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt < self.max_retries:
                    print(
                        f"[WARNING] Retriever timeout/error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. Retrying in {self.retry_backoff}s "
                        f"(is the retriever server at {self.url} running?)...",
                        file=sys.stderr,
                    )
                    time.sleep(self.retry_backoff)
                else:
                    print(
                        f"[ERROR] Retriever failed after {self.max_retries + 1} attempts: "
                        f"{e}. Is the server at {self.url} running?",
                        file=sys.stderr,
                    )
                    raise

    # The client is a cheap, stateless handle over a shared session -- it is safe to
    # share across program copies, so deepcopy/copy return self (matching the old
    # retriever's contract that the optimizer never clones retrieval infrastructure).
    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self


_default_retriever = None
_retriever_lock = threading.Lock()


def get_default_retriever() -> HTTPEmbeddingRetriever:
    """Return the process-wide retriever client singleton.

    Now near-instant to construct (no index load), so importing the benchmark/pipeline
    no longer blocks on multiple GB of IO. Double-checked locking keeps a single shared
    client even under concurrent first-calls from worker threads.
    """
    global _default_retriever
    if _default_retriever is None:
        with _retriever_lock:
            if _default_retriever is None:
                _default_retriever = HTTPEmbeddingRetriever()
    return _default_retriever
