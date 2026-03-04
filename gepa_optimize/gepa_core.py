"""
Program-agnostic GEPA orchestration: compile, save, evaluate on val, save results.

Callers are responsible for: LM configuration (student LM via dspy.configure),
dataset loading, program instantiation, metric, and preflight checks.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
from dspy.evaluate import Evaluate


def run_gepa_optimization(
    program: dspy.Module,
    train_set: list,
    val_set: list,
    metric: Any,
    output_dir: Path,
    args: Any,
) -> tuple[dspy.Module, float]:
    """Run GEPA compile, save optimized program, evaluate on val set. Returns (optimized_program, val_score)."""
    # Disable SQLite disk cache to avoid contention with multiple threads.
    dspy.cache.enable_disk_cache = False

    reflection_lm = dspy.LM(args.reflection_lm, temperature=1.0, max_tokens=16000)

    print(f"\nStarting GEPA optimization (auto={args.auto}, seed={args.seed})...")
    gepa = dspy.GEPA(
        metric=metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
        seed=args.seed,
        log_dir=str(output_dir / "gepa_logs"),
        track_stats=True,
    )

    optimized_program = gepa.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    # Save optimized program
    # Workaround: DSPy's Retrieve.dump_state() doesn't accept the json_mode kwarg
    # that BaseModule.dump_state() passes. Patch it before saving.
    from dspy.retrievers.retrieve import Retrieve as _Retrieve

    _orig_dump = _Retrieve.dump_state
    if "json_mode" not in inspect.signature(_orig_dump).parameters:
        _Retrieve.dump_state = lambda self, **kwargs: _orig_dump(self)

    save_path = output_dir / "gepa_optimized_program.json"
    optimized_program.save(str(save_path))

    _Retrieve.dump_state = _orig_dump  # restore
    print(f"\nSaved optimized program to {save_path}")

    # Evaluate on val set
    print("\nEvaluating optimized program on val set...")
    evaluator = Evaluate(
        devset=val_set,
        metric=metric,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=5000,
    )
    val_result = evaluator(optimized_program)
    val_score_value = (
        float(val_result.score) if hasattr(val_result, "score") else float(val_result)
    )
    print(f"Final val score: {val_score_value}")

    return optimized_program, val_score_value


def save_gepa_results(
    output_dir: Path,
    val_score: float,
    args: Any,
    train_size: int,
    val_size: int,
) -> None:
    """Write results.json and results.txt."""
    results = {
        "val_score": val_score,
        "args": vars(args),
        "train_size": train_size,
        "val_size": val_size,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    results_txt_path = output_dir / "results.txt"
    with open(results_txt_path, "w") as f:
        f.write("GEPA Optimization Results\n")
        f.write("========================\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Val score: {val_score}\n")
        f.write(f"Auto budget: {args.auto}\n")
        f.write(f"Student LM: {args.lm}\n")
        f.write(f"Reflection LM: {args.reflection_lm}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Train size: {train_size}\n")
        f.write(f"Val size: {val_size}\n")
        if getattr(args, "program_path", None):
            f.write(f"Starting program: {args.program_path}\n")

    print(f"Saved results to {results_path} and {results_txt_path}")
