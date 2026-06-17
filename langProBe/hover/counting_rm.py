import sys
import threading

import requests


class CountingRM:
    """Wraps any retrieval model to count queries. Thread-safe for parallel evaluation.

    Counting uses ``threading.local()`` so each evaluation thread (from
    dspy.Evaluate's thread pool) tracks its own query count independently --
    essential when the pipeline runs with num_threads > 1.

    It also raises ColBERTv2's underlying request timeout above the 10s default
    and retries transient timeout/connection errors, which the remote Modal
    ColBERT server is prone to under concurrent load.

    Usage:
        rm = CountingRM(dspy.ColBERTv2(url=...))
        rm.reset_count()
        # ... run pipeline with dspy.context(rm=rm) ...
        num_retrievals = rm.get_count()
    """

    def __init__(self, rm, timeout=60, max_retries=2):
        self._rm = rm
        self._local = threading.local()
        self.timeout = timeout
        self.max_retries = max_retries
        # Override the default 10s timeout in ColBERTv2's underlying requests.
        self._patch_timeout()

    def _patch_timeout(self):
        """Monkey-patch the ColBERTv2 GET request function to use our timeout."""
        import dspy.dsp.colbertv2 as colbert_mod

        timeout = self.timeout

        def patched_get(url, query, k):
            payload = {"query": query, "k": k}
            res = requests.get(url, params=payload, timeout=timeout)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("error"):
                raise ValueError(
                    f"ColBERTv2 server returned an error: {res_json.get('message', 'Unknown error')}"
                )
            if "topk" not in res_json:
                raise ValueError(
                    f"ColBERTv2 server returned an unexpected response: {res_json}"
                )
            topk = res_json["topk"][:k]
            topk = [{**d, "long_text": d["text"]} for d in topk]
            return topk[:k]

        colbert_mod.colbertv2_get_request_v2 = patched_get
        colbert_mod.colbertv2_get_request = patched_get

    def __call__(self, *args, **kwargs):
        self._local.count = getattr(self._local, "count", 0) + 1
        for attempt in range(self.max_retries + 1):
            try:
                return self._rm(*args, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    print(
                        f"[WARNING] Retrieval timeout/error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. Retrying...",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[ERROR] Retrieval failed after {self.max_retries + 1} attempts: {e}",
                        file=sys.stderr,
                    )
                    raise

    def reset_count(self):
        self._local.count = 0

    def get_count(self):
        return getattr(self._local, "count", 0)
