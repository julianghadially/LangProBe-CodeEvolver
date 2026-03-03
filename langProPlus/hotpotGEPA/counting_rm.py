import threading


class CountingRM:
    """Wraps any retrieval model to count queries. Thread-safe for parallel evaluation.

    Uses threading.local() so each evaluation thread (from dspy.Evaluate's
    thread pool) tracks its own count independently.

    Usage:
        rm = CountingRM(dspy.ColBERTv2(url=...))
        rm.reset_count()
        # ... run pipeline with dspy.context(rm=rm) ...
        num_retrievals = rm.get_count()
    """

    def __init__(self, rm):
        self._rm = rm
        self._local = threading.local()

    def __call__(self, *args, **kwargs):
        self._local.count = getattr(self._local, "count", 0) + 1
        return self._rm(*args, **kwargs)

    def reset_count(self):
        self._local.count = 0

    def get_count(self):
        return getattr(self._local, "count", 0)
