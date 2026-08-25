"""
Transparent evaluation pipeline for HotpotQA with per-example logging.

Usage:
    # Baseline on test set
    python -m simple_eval.evaluate --split test

    # GEPA-optimized program on test set. Generate the JSON with
    # `python -m gepa_optimize.run_gepa --program hotpot`; its output directory
    # is git-ignored, so no optimized program ships in the repo.
    python -m simple_eval.evaluate --split test \
        --program_path gepa_optimize/output_<program>_<timestamp>/gepa_optimized_program.json

    # Reproduce GEPA val subsample (seed=42, n=150)
    python -m simple_eval.evaluate --split val --seed 42 --n 150 \
        --program_path gepa_optimize/output_<program>_<timestamp>/gepa_optimized_program.json
"""

import argparse
from pathlib import Path

import dspy

from . import core
from .programs.hotpot import (  # type: ignore[reportMissingImports]
    check_dependencies_hotpot,
    load_dataset_hotpot,
    load_program_hotpot,
    select_metric_hotpot,
)

def main():
    parser = argparse.ArgumentParser(
        description="Transparent HotpotQA evaluation with per-example logging"
    )
    parser.add_argument("--program_path", type=str, default=None,
                        help="Path to saved program JSON (baseline if omitted)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "val", "test"],
                        help="Dataset split (default: test)")
    parser.add_argument("--n", type=int, default=None,
                        help="Evaluate only first N examples (or subsample N with --seed)")
    parser.add_argument("--lm", type=str, default="openai/gpt-4.1-mini",
                        help="Language model (default: openai/gpt-4.1-mini)")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="Parallel eval threads (default: 8)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-timestamped if omitted)")
    parser.add_argument("--experiment_name", type=str, default="simple_eval_hotpot",
                        help="MLflow experiment name")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Disable MLflow tracing")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shuffle seed; with --n, uses random.Random(seed).sample()")
    parser.add_argument("--metric", type=str, default=None,
                        choices=["exact_match", "resource_penalty", "llm_judge"],
                        help="Metric to use: exact_match (default), resource_penalty, or llm_judge")
    args = parser.parse_args()

    # Select metric (Hotpot-specific).
    metric, metric_name = select_metric_hotpot(args.metric)

    # Build config dict for logging
    config = {
        "program_path": args.program_path or "baseline",
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "lm": args.lm,
        "num_threads": args.num_threads,
        "metric": metric_name,
    }

    # Output directory
    label = Path(args.program_path).stem if args.program_path else "baseline"
    output_dir = core.build_output_dir(label, args.split, args.output_dir)

    print(f"=== simple_eval ===")
    print(f"Split: {args.split}, N: {args.n}, Seed: {args.seed}")
    print(f"Program: {args.program_path or 'baseline'}")
    print(f"Metric: {metric_name}")
    print(f"LM: {args.lm}")
    print(f"Output: {output_dir}")
    print()

    # Disable all DSPy caching for clean evaluation
    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass

    # Configure LM
    lm = dspy.LM(args.lm)
    dspy.configure(lm=lm, experimental=True)

    # ColBERT preflight (Hotpot-specific dependency check).
    check_dependencies_hotpot()

    # Load dataset (Hotpot)
    print(f"\nLoading {args.split} set...")
    dataset = load_dataset_hotpot(args.split, args.n, args.seed)
    print(f"Loaded {len(dataset)} examples")

    # Load program (Hotpot)
    print(f"\nLoading program...")
    program = load_program_hotpot(args.program_path)

    # MLflow setup
    mlflow_run = None
    if not args.no_mlflow:
        mlflow_run = core.setup_mlflow(args.experiment_name, config)

    # Run evaluation using shared core harness.
    print(f"\nRunning evaluation ({args.num_threads} threads)...")
    overall_score, per_example = core.run_evaluation(
        program, dataset, args.num_threads, metric=metric
    )
    print(f"\n{'='*40}")
    print(f"Overall score: {overall_score:.2f}%")
    print(f"{'='*40}")

    # Save results
    core.save_results(output_dir, overall_score, per_example, config)

    # MLflow finalize
    core.finalize_mlflow(mlflow_run, overall_score, output_dir)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
