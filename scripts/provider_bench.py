"""Benchmark one inference provider on Hover: throughput, latency, score.

Runs the real ``HoverMultiHopPipeline`` over a sample of rows with a thread
pool, pinned to a SINGLE provider (the cross-provider fallback in
``langProBe.lm_provider`` is switched off), and records per-row and per-LM-call
timings so two arms can be compared directly.

Why not ``simple_eval.evaluate_hover``: ``dspy.Evaluate`` reports only the
aggregate score. The question here is a latency-distribution one -- DeepInfra's
DeepSeek deploy was previously slow, and its characteristic failure is a single
hanging request that stalls one row far past the rest. That shows up in the
tail (p95/max, and per-LM-call max), not in the mean, and not in the score.

Usage:
    python scripts/provider_bench.py --provider gmi       --n 200 --threads 25
    python scripts/provider_bench.py --provider deepinfra --n 200 --threads 25

Both arms take the same ``--seed``, so they evaluate the identical rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["gmi", "deepinfra", "deepseek"], required=True)
    p.add_argument("--n", type=int, default=200, help="rows to evaluate (default 200)")
    p.add_argument("--threads", type=int, default=25, help="concurrent rows (default 25)")
    p.add_argument(
        "--split",
        default="trainval",
        help="data/hoverBench_<split>.json (default trainval -- the 450-row "
             "train+val pool CodeEvolver optimizes against)",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="row-sample seed; use the same value for both arms")
    p.add_argument("--lm-timeout", type=float, default=None,
                   help="per-LM-request timeout in seconds (default: none, so a "
                        "hanging request is measured rather than cut off)")
    p.add_argument("--max-wall", type=float, default=None,
                   help="stop waiting after this many seconds and report the "
                        "still-running rows as unfinished")
    p.add_argument("--heartbeat", type=float, default=60.0,
                   help="seconds between progress lines (default 60)")
    p.add_argument("--out", default=None, help="output JSON path")
    return p.parse_args()


args = parse_args()

# MUST precede the pipeline import: `build_task_lm()` reads these at construction.
os.environ["LM_PROVIDER"] = args.provider
os.environ["LM_FALLBACK"] = "0"  # a single-provider measurement stays single-provider

import dspy  # noqa: E402
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline  # noqa: E402
from langProBe.hover.hover_utils import discrete_retrieval_eval  # noqa: E402
from langProBe.lm_provider import build_task_lm  # noqa: E402


# --- per-LM-call instrumentation ------------------------------------------
#
# `lm.history` is shared across threads and cannot be attributed to a row, so
# wrap `forward` and stash each call's duration on a thread-local list that the
# row worker drains. This separates "the provider is slow per call" from "the
# program made more calls".

_calls = threading.local()


def _usage(response) -> dict:
    """Token counts off a LiteLLM ModelResponse; {} if the shape is unexpected."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    det = getattr(u, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
    }


def instrument(lm: dspy.LM) -> dspy.LM:
    """Record duration AND token counts per call.

    Duration alone cannot tell a slow call from a long one. Pairing it with
    completion tokens separates the two mechanisms outright: a slow call that
    emitted proportionally more tokens is the model reasoning longer, while one
    that emitted a normal number of tokens over a long wall time was stalled --
    queued, throttled, or retried underneath us.
    """
    inner = lm.forward

    def forward(prompt=None, messages=None, **kwargs):
        started = time.monotonic()
        record = {"retries": _retry_counter.take()}
        try:
            response = inner(prompt=prompt, messages=messages, **kwargs)
            record.update(_usage(response))
            return response
        except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised
            record["error"] = type(exc).__name__
            raise
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)
            # Retries logged DURING this call, not before it.
            record["retries"] = _retry_counter.take()
            if not hasattr(_calls, "records"):
                _calls.records = []
            _calls.records.append(record)

    lm.forward = forward
    return lm


class _RetryCounter(logging.Handler):
    """Counts LiteLLM's own retry/rate-limit log lines, per thread.

    LiteLLM retries 429/5xx internally and transparently, so a throttled request
    surfaces to us only as a long duration -- indistinguishable from slow
    generation unless we watch its logger. Attached to the LiteLLM loggers at
    the level they emit retry chatter on.
    """

    # Match retry EVENTS only. A bare "retry" substring also hits LiteLLM's
    # per-call parameter dump ("...num_retries=3, retry_strategy=..."), which is
    # logged on every clean call and made an earlier version of this counter
    # report one retry per request. Anything naming the config is excluded.
    KEYWORDS = ("retrying request", "retry attempt", "rate limit", "ratelimit",
                "429", "too many requests", "overloaded")
    EXCLUDE = ("num_retries=", "retry_strategy=", "retry_policy")

    def __init__(self):
        super().__init__(level=logging.INFO)
        self._local = threading.local()

    def emit(self, record):
        try:
            msg = record.getMessage().lower()
        except Exception:  # noqa: BLE001 -- logging must never break a call
            return
        if any(x in msg for x in self.EXCLUDE):
            return
        if any(k in msg for k in self.KEYWORDS):
            self._local.n = getattr(self._local, "n", 0) + 1

    def take(self) -> int:
        n = getattr(self._local, "n", 0)
        self._local.n = 0
        return n


_retry_counter = _RetryCounter()


def arm_retry_counter() -> None:
    # openai._base_client is where the actual "Retrying request ... in Ns" lines
    # come from; the LiteLLM loggers carry rate-limit chatter.
    for name in ("LiteLLM", "litellm", "LiteLLM Router", "openai._base_client"):
        lg = logging.getLogger(name)
        lg.addHandler(_retry_counter)
        # INFO, never DEBUG: the real "Retrying request ... in Ns" lines are INFO,
        # while DEBUG turns LiteLLM into a per-request firehose that would bloat
        # the log and perturb the very timings being measured.
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)


# --- run state -------------------------------------------------------------

_lock = threading.Lock()
_results: list[dict] = []
_inflight: dict[int, float] = {}   # row idx -> monotonic start
_done = threading.Event()


def run_row(program, idx: int, example) -> None:
    _calls.records = []
    started = time.monotonic()
    with _lock:
        _inflight[idx] = started
    row: dict = {"idx": idx, "claim": example.claim}
    try:
        pred = program(claim=example.claim)
        row["score"] = float(discrete_retrieval_eval(example, pred))
        row["search_count"] = int(getattr(pred, "search_count", 0) or 0)
        row["error"] = None
    except Exception as exc:  # noqa: BLE001 -- a failed row is a data point
        row["score"] = 0.0
        row["search_count"] = None
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"
    finally:
        row["seconds"] = time.monotonic() - started
        records = list(getattr(_calls, "records", []))
        durations = [r["seconds"] for r in records]
        row["lm_calls"] = len(records)
        row["lm_seconds_total"] = round(sum(durations), 3)
        row["lm_seconds_max"] = round(max(durations), 3) if durations else None
        row["lm_call_seconds"] = durations
        row["lm_call_records"] = records
        row["lm_retries"] = sum(r.get("retries", 0) for r in records)
        row["seconds"] = round(row["seconds"], 3)
        with _lock:
            _inflight.pop(idx, None)
            _results.append(row)


def heartbeat(total: int, t0: float, every: float) -> None:
    """Name the in-flight rows and how long each has been running -- a hanging
    row is visible while it hangs, not only in the post-hoc percentiles."""
    while not _done.wait(every):
        now = time.monotonic()
        with _lock:
            done_n = len(_results)
            waiting = sorted(((now - s, i) for i, s in _inflight.items()), reverse=True)
        elapsed = now - t0
        head = ", ".join(f"row{i}@{d:.0f}s" for d, i in waiting[:5])
        print(f"[{elapsed:7.0f}s] {done_n}/{total} done, {len(waiting)} in flight"
              + (f" | oldest: {head}" if head else ""), flush=True)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; `statistics.quantiles` needs n>=2 and interpolates."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q / 100 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def _tok_per_sec_summary(ok_rows: list[dict]) -> dict:
    """Completion-token throughput, split at the p95 call duration.

    If the slow tail generates tokens at roughly the same rate as the fast body,
    the slow calls were simply longer generations. If the tail's rate collapses,
    those calls spent their time waiting rather than generating.
    """
    recs = [r for row in ok_rows for r in row.get("lm_call_records", [])
            if r.get("completion_tokens") and r.get("seconds")]
    if not recs:
        return {}
    cut = percentile([r["seconds"] for r in recs], 95)
    fast = [r for r in recs if r["seconds"] < cut]
    slow = [r for r in recs if r["seconds"] >= cut]

    def rate(rs):
        if not rs:
            return None
        return round(sum(r["completion_tokens"] for r in rs) / sum(r["seconds"] for r in rs), 1)

    def toks(rs):
        return round(statistics.fmean([r["completion_tokens"] for r in rs]), 0) if rs else None

    return {
        "cut_seconds": round(cut, 1),
        "fast_calls": len(fast), "fast_tok_per_s": rate(fast), "fast_mean_tokens": toks(fast),
        "slow_calls": len(slow), "slow_tok_per_s": rate(slow), "slow_mean_tokens": toks(slow),
    }


def main() -> int:
    primary = args.provider

    path = PROJECT_ROOT / "data" / f"hoverBench_{args.split}.json"
    raw = json.loads(path.read_text())
    examples = [
        dspy.Example(claim=ex["claim"], supporting_facts=ex["supporting_facts"],
                     label=ex["label"]).with_inputs("claim")
        for ex in raw
    ]
    n = min(args.n, len(examples))
    if n < args.n:
        print(f"WARNING: requested n={args.n} but {path.name} has {len(examples)} rows")
    sample = random.Random(args.seed).sample(examples, n)

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "scripts" / "provider_bench_results"
        / f"hover_{primary}_n{n}_t{args.threads}_"
          f"{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    program = HoverMultiHopPipeline()
    lm_overrides = {"timeout": args.lm_timeout} if args.lm_timeout else {}
    if lm_overrides:
        program.lm = build_task_lm(**lm_overrides)
    assert program.lm._fallback is None, (
        "single-provider arm must have the cross-provider fallback disarmed")
    arm_retry_counter()
    program.lm = instrument(program.lm)

    print(f"=== provider_bench: hover / {primary} ===")
    print(f"model={program.lm.model} fallback=disabled")
    print(f"rows={n} from {path.name} (seed={args.seed}) threads={args.threads} "
          f"lm_timeout={args.lm_timeout}")
    print(f"out={out_path}\n", flush=True)

    t0 = time.monotonic()
    hb = threading.Thread(target=heartbeat, args=(n, t0, args.heartbeat), daemon=True)
    hb.start()

    # A plain pool of daemon workers rather than ThreadPoolExecutor: --max-wall
    # must be able to abandon a hung request, and a non-daemon pool thread stuck
    # in a socket read keeps the interpreter alive at exit.
    queue = list(enumerate(sample))
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue:
                    return
                idx, example = queue.pop(0)
            run_row(program, idx, example)

    workers = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(args.threads, n))]
    for w in workers:
        w.start()

    deadline = t0 + args.max_wall if args.max_wall else None
    for w in workers:
        w.join(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
    _done.set()

    wall = time.monotonic() - t0
    with _lock:
        rows = sorted(_results, key=lambda r: r["idx"])
        unfinished = sorted(_inflight)

    ok = [r for r in rows if r["error"] is None]
    times = [r["seconds"] for r in ok]
    call_times = [d for r in ok for d in r["lm_call_seconds"]]
    summary = {
        "provider": primary,
        "model": program.lm.model,
        "split": args.split,
        "seed": args.seed,
        "threads": args.threads,
        "requested_rows": n,
        "completed_rows": len(rows),
        "unfinished_rows": unfinished,
        "errored_rows": len(rows) - len(ok),
        "wall_seconds": round(wall, 1),
        "rows_per_minute": round(len(rows) / wall * 60, 2) if wall else None,
        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None,
        "row_seconds": {
            "mean": round(statistics.fmean(times), 1) if times else None,
            "p50": round(percentile(times, 50), 1) if times else None,
            "p90": round(percentile(times, 90), 1) if times else None,
            "p95": round(percentile(times, 95), 1) if times else None,
            "max": round(max(times), 1) if times else None,
        },
        "lm_call_seconds": {
            "count": len(call_times),
            "mean": round(statistics.fmean(call_times), 2) if call_times else None,
            "p50": round(percentile(call_times, 50), 2) if call_times else None,
            "p90": round(percentile(call_times, 90), 2) if call_times else None,
            "p99": round(percentile(call_times, 99), 2) if call_times else None,
            "max": round(max(call_times), 2) if call_times else None,
        },
        "lm_retries_total": sum(r.get("lm_retries", 0) for r in rows),
        "tokens_per_second": _tok_per_sec_summary(ok),
        "lm_calls_per_row_mean": round(statistics.fmean([r["lm_calls"] for r in ok]), 1) if ok else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    slowest = sorted(ok, key=lambda r: -r["seconds"])[:5]
    print("slowest rows: " + ", ".join(
        f"row{r['idx']} {r['seconds']:.0f}s ({r['lm_calls']} calls, "
        f"max call {r['lm_seconds_max']:.0f}s)" for r in slowest))
    errs = [r for r in rows if r["error"]]
    if errs:
        print(f"\n{len(errs)} errored rows:")
        for r in errs[:10]:
            print(f"  row{r['idx']} after {r['seconds']:.0f}s: {r['error'][:200]}")
    print(f"\nWrote {out_path}")

    # Daemon workers may still be blocked in a socket read; results are on disk.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
