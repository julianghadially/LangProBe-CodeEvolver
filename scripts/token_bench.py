"""Token benchmark for Hover multi-hop pipeline: DeepSeek-V4-Flash vs GPT-5.4-nano.

Runs N test rows through the exact HoverMultiHopPipeline and sums per-call token
usage from dspy.LM.history, including reasoning tokens.

Usage:
    python token_bench.py <deepseek|nano> <n_rows> <num_threads>
"""

from __future__ import annotations

import sys
import time

import dspy

from simple_eval.programs.hover import (
    check_dependencies_hover,
    load_dataset_hover,
    select_metric_hover,
)
from simple_eval import core
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline


def _g(obj, key, default=0):
    """Get key from a dict or attr from an object; treat None/missing as default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return default if val is None else val


def summarize_usage(history):
    """Sum usage across all LM calls in history."""
    n_calls = 0
    prompt = 0          # input tokens
    completion = 0      # output tokens (INCLUDES reasoning for OpenAI-style)
    reasoning = 0       # reasoning tokens (subset of completion)
    total = 0
    cost = 0.0
    cost_available = False
    for entry in history:
        usage = entry.get("usage") if isinstance(entry, dict) else None
        if not usage:
            continue
        n_calls += 1
        p = int(_g(usage, "prompt_tokens", 0))
        c = int(_g(usage, "completion_tokens", 0))
        t = int(_g(usage, "total_tokens", p + c))
        details = _g(usage, "completion_tokens_details", None)
        r = int(_g(details, "reasoning_tokens", 0))
        prompt += p
        completion += c
        reasoning += r
        total += t
        entry_cost = entry.get("cost") if isinstance(entry, dict) else None
        if entry_cost is not None:
            cost += float(entry_cost)
            cost_available = True
    return {
        "n_calls": n_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "cost_usd": cost if cost_available else None,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nano"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    # Hard-disable caching so every call hits the real LM (matches production).
    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass

    program = HoverMultiHopPipeline()

    if mode == "deepseek":
        # Use the pipeline's own LM verbatim (deepinfra/deepseek-ai/DeepSeek-V4-Flash,
        # reasoning_effort="high") -- the exact production config.
        model_desc = "pipeline default: deepinfra/deepseek-ai/DeepSeek-V4-Flash (reasoning_effort=high)"
    elif mode == "nano":
        program.lm = dspy.LM(
            "openai/gpt-5.4-nano",
            reasoning_effort="low",
        )
        model_desc = "openai/gpt-5.4-nano (reasoning_effort=low)"
    elif mode == "gmi":
        # Same model (DeepSeek-V4-Flash) and same reasoning_effort as the
        # DeepInfra path -- only the serving provider changes, via LiteLLM's
        # OpenAI-compatible route (openai/<model> + api_base).
        import os
        program.lm = dspy.LM(
            "openai/deepseek-ai/DeepSeek-V4-Flash",
            api_base="https://api.gmi-serving.com/v1",
            api_key=os.environ["GMI_API_KEY"],
            reasoning_effort="high",
            allowed_openai_params=["reasoning_effort"],
        )
        model_desc = "GMI openai/deepseek-ai/DeepSeek-V4-Flash @ api.gmi-serving.com (reasoning_effort=high)"
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'deepseek' or 'nano'")

    lm = program.lm
    lm.history.clear()

    metric, metric_name = select_metric_hover(None)

    print(f"=== token_bench mode={mode} n={n_rows} threads={num_threads} ===", flush=True)
    print(f"LM: {model_desc}", flush=True)
    check_dependencies_hover()

    # split="test", seed=None, n=n_rows -> first n_rows examples (identical rows
    # for both models, deterministic).
    dataset = load_dataset_hover("test", n_rows, None)
    print(f"Loaded {len(dataset)} rows", flush=True)

    start = time.time()
    overall_score, _ = core.run_evaluation(program, dataset, num_threads, metric=metric)
    elapsed = time.time() - start

    stats = summarize_usage(lm.history)
    rows = len(dataset)

    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS  mode={mode}  rows={rows}  threads={num_threads}", flush=True)
    print("=" * 60, flush=True)
    print(f"retrieval_score      : {overall_score:.4f}", flush=True)
    print(f"wall_time_seconds    : {elapsed:.1f}", flush=True)
    print(f"lm_calls             : {stats['n_calls']}  ({stats['n_calls']/rows:.2f} per row)", flush=True)
    print(f"input_tokens         : {stats['prompt_tokens']:,}", flush=True)
    print(f"output_tokens        : {stats['completion_tokens']:,}   (includes reasoning)", flush=True)
    print(f"  of which reasoning : {stats['reasoning_tokens']:,}", flush=True)
    print(f"total_tokens         : {stats['total_tokens']:,}", flush=True)
    if stats["cost_usd"] is not None:
        print(f"litellm_cost_usd     : ${stats['cost_usd']:.4f}", flush=True)
    else:
        print(f"litellm_cost_usd     : (not reported by provider)", flush=True)
    print(f"tokens_per_row       : {stats['total_tokens']/rows:,.0f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
