"""
Transparent evaluation pipeline for RAGQAArenaTech with per-example logging.

Usage:
    # Baseline (SimplifiedBaleen) on a 5-example test subsample
    python -m simple_eval.evaluate_ragqa --split test --n 5 --no_mlflow

    # Full test split
    python -m simple_eval.evaluate_ragqa --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy

from . import core
from .programs.ragqa import (  # type: ignore[reportMissingImports]
    check_dependencies_ragqa,
    load_dataset_ragqa,
    load_program_ragqa,
    select_metric_ragqa,
)


def main():
    parser = argparse.ArgumentParser(
        description="Transparent RAGQAArenaTech evaluation with per-example logging"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default=None,
        help="Path to saved program JSON (reserved for future optimized RAGQAArenaTech programs)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "val", "test"],
        help="Dataset split. RAGQAArenaTech has no dev set; 'dev' aliases 'val'.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only first N examples (or subsample N with --seed)",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Task language model (default: openai/gpt-4.1-mini)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Parallel eval threads (default: 8)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-timestamped if omitted)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="simple_eval_ragqa",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow tracing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed; with --n, uses random.Random(seed).sample()",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["semantic_f1"],
        help="Metric to use (currently only: semantic_f1)",
    )
    args = parser.parse_args()

    # Select metric (RAGQAArenaTech-specific).
    metric, metric_name = select_metric_ragqa(args.metric)

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
    label = Path(args.program_path).stem if args.program_path else "ragqa_baseline"
    output_dir = core.build_output_dir(label, args.split, args.output_dir)

    print("=== simple_eval (RAGQAArenaTech) ===")
    print(f"Split: {args.split}, N: {args.n}, Seed: {args.seed}")
    print(f"Program: {args.program_path or 'baseline RAGQAArenaTechPipeline (SimplifiedBaleen)'}")
    print(f"Metric: {metric_name}")
    print(f"LM: {args.lm}")
    print(f"Output: {output_dir}")
    print()

    # Disable all DSPy caching for clean evaluation
    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass

    # Configure task LM
    lm = dspy.LM(args.lm)
    dspy.configure(lm=lm, experimental=True)

    # Dependency preflight (RAGQAArenaTech-specific).
    check_dependencies_ragqa()

    # Load dataset (RAGQAArenaTech)
    print(f"\nLoading {args.split} RAGQAArenaTech split...")
    dataset = load_dataset_ragqa(args.split, args.n, args.seed)
    print(f"Loaded {len(dataset)} RAGQAArenaTech examples")

    # Load program (RAGQAArenaTech) -- triggers corpus + index download on first run
    print(f"\nLoading RAGQAArenaTech program...")
    program = load_program_ragqa(args.program_path)

    # MLflow setup
    mlflow_run = None
    if not args.no_mlflow:
        mlflow_run = core.setup_mlflow(args.experiment_name, config)

    # Run evaluation using shared core harness.
    print(f"\nRunning RAGQAArenaTech evaluation ({args.num_threads} threads)...")
    overall_score, per_example = core.run_evaluation(
        program, dataset, args.num_threads, metric=metric
    )
    print(f"\n{'='*40}")
    print(f"Overall RAGQAArenaTech SemanticF1 score: {overall_score:.2f}")
    print(f"{'='*40}")

    # Save results
    core.save_results(output_dir, overall_score, per_example, config)

    # MLflow finalize
    core.finalize_mlflow(mlflow_run, overall_score, output_dir)

    print(f"\nDone. RAGQAArenaTech results in {output_dir}/")


if __name__ == "__main__":
    main()
