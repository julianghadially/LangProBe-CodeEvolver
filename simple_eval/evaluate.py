"""
Transparent evaluation pipeline for HotpotQA with per-example logging.

Usage:
    # Baseline on test set
    python -m simple_eval.evaluate --split test

    # GEPA-optimized program on test set
    python -m simple_eval.evaluate --split test \
        --program_path gepa_optimize/output_promptonly_gepa/gepa_optimized_program.json

    # Reproduce GEPA val subsample (seed=42, n=150)
    python -m simple_eval.evaluate --split val --seed 42 --n 150 \
        --program_path gepa_optimize/output_promptonly_gepa/gepa_optimized_program.json
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import requests as _requests

import dspy
from dspy.evaluate import Evaluate, answer_exact_match

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(split: str, n: int | None, seed: int | None) -> list[dspy.Example]:
    """Load HotpotQA examples from a JSON split file.

    When both seed and n are provided, uses random.Random(seed).sample(examples, n)
    to reproduce the same subsampling used during GEPA optimization.
    """
    path = PROJECT_ROOT / "data" / f"HotpotQABench_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    examples = [
        dspy.Example(
            question=ex["question"],
            answer=ex["answer"],
            gold_titles=ex.get("gold_titles", []),
        ).with_inputs("question")
        for ex in raw
    ]

    if seed is not None and n is not None:
        if n > len(examples):
            print(f"WARNING: requested n={n} but only {len(examples)} examples available")
            n = len(examples)
        examples = random.Random(seed).sample(examples, n)
    elif n is not None:
        examples = examples[:n]

    return examples


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

def load_program(program_path: str | None) -> dspy.Module:
    """Create HotpotMultiHopPipeline, optionally loading saved state."""
    from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPipeline

    program = HotpotMultiHopPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        program.load(str(path))
        print(f"Loaded program state from {path}")
    return program


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _extract_score(score) -> float:
    """Extract a float from a metric return value (handles ScoreWithFeedback)."""
    if hasattr(score, "score"):
        return float(score.score)
    return float(score)


def run_evaluation(
    program: dspy.Module,
    dataset: list[dspy.Example],
    num_threads: int,
    metric=None,
) -> tuple[float, list[dict]]:
    """Run dspy.evaluate.Evaluate and extract per-example results.

    Returns (overall_score, per_example_list) where each entry in
    per_example_list is a dict with idx, question, gold_answer,
    predicted_answer, score, gold_titles, and optionally retrieval_count.
    """
    if metric is None:
        metric = answer_exact_match

    evaluator = Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=5000,
    )
    eval_result = evaluator(program)

    # EvaluationResult has .score (float) and .results (list of (example, prediction, score))
    overall_score = eval_result.score
    if hasattr(overall_score, "score"):
        overall_score = overall_score.score
    results_list = eval_result.results

    per_example = []
    for idx, (example, prediction, score) in enumerate(results_list):
        row = {
            "idx": idx,
            "question": example.question,
            "gold_answer": example.answer,
            "predicted_answer": getattr(prediction, "answer", str(prediction)),
            "score": _extract_score(score),
            "gold_titles": json.dumps(example.get("gold_titles", [])),
        }
        retrieval_count = getattr(prediction, "retrieval_count", None)
        if retrieval_count is not None:
            row["retrieval_count"] = int(retrieval_count)
        per_example.append(row)

    return overall_score, per_example


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    overall_score: float,
    per_example: list[dict],
    config: dict,
) -> None:
    """Write per_example_results.csv and summary.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "per_example_results.csv"
    fieldnames = ["idx", "question", "gold_answer", "predicted_answer", "score", "gold_titles"]
    if per_example and "retrieval_count" in per_example[0]:
        fieldnames.append("retrieval_count")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_example)
    print(f"Saved per-example results to {csv_path}")

    # Summary JSON
    correct = sum(1 for row in per_example if row["score"] > 0)
    total = len(per_example)
    summary = {
        "overall_score": overall_score,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")


# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_name: str, config: dict, tracking_uri: str = "http://127.0.0.1:5000"):
    """Set up MLflow experiment and autologging. Returns run or None."""
    try:
        import mlflow
        import mlflow.dspy
    except ImportError:
        print("mlflow not installed, skipping MLflow integration")
        return None

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_experiment(experiment_name)
    mlflow.dspy.autolog(log_traces=True, log_traces_from_eval=True)
    run = mlflow.start_run()
    mlflow.log_params({k: str(v) for k, v in config.items()})
    print(f"MLflow run started: {run.info.run_id}")
    return run


def finalize_mlflow(run, overall_score: float, output_dir: Path):
    """Log metrics and artifacts, then end the MLflow run."""
    if run is None:
        return
    try:
        import mlflow

        mlflow.log_metric("overall_score", overall_score)
        csv_path = output_dir / "per_example_results.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path))
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            mlflow.log_artifact(str(summary_path))
        mlflow.end_run()
        print(f"MLflow run finalized: {run.info.run_id}")
    except Exception as e:
        print(f"WARNING: MLflow finalization failed: {e}")


# ---------------------------------------------------------------------------
# ColBERT preflight
# ---------------------------------------------------------------------------

def check_colbert():
    """Verify ColBERT retrieval server is reachable."""
    from langProPlus.hotpotGEPA.hotpot_pipeline import COLBERT_URL

    print(f"Checking ColBERT server at {COLBERT_URL}...")
    try:
        resp = _requests.get(COLBERT_URL, params={"query": "test", "k": 1}, timeout=15)
        resp.raise_for_status()
        print(f"ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: ColBERT server unreachable: {e}")
        print("The pipeline requires ColBERT for retrieval. Proceeding anyway, but expect errors.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--resource_metric", action="store_true",
                        help="Use composite accuracy + retrieval penalty metric (0.02/query)")
    args = parser.parse_args()

    # Select metric
    if args.resource_metric:
        from langProPlus.hotpotGEPA.hotpot_metric_resource import (
            hotpot_accuracy_with_resource_penalty_feedback,
        )
        metric = hotpot_accuracy_with_resource_penalty_feedback
        metric_name = "accuracy_with_resource_penalty"
    else:
        metric = answer_exact_match
        metric_name = "answer_exact_match"

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
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        label = Path(args.program_path).stem if args.program_path else "baseline"
        output_dir = Path(f"simple_eval/results/{label}_{args.split}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # ColBERT preflight
    check_colbert()

    # Load dataset
    print(f"\nLoading {args.split} set...")
    dataset = load_dataset(args.split, args.n, args.seed)
    print(f"Loaded {len(dataset)} examples")

    # Load program
    print(f"\nLoading program...")
    program = load_program(args.program_path)

    # MLflow setup
    mlflow_run = None
    if not args.no_mlflow:
        mlflow_run = setup_mlflow(args.experiment_name, config)

    # Run evaluation
    print(f"\nRunning evaluation ({args.num_threads} threads)...")
    overall_score, per_example = run_evaluation(program, dataset, args.num_threads, metric=metric)
    print(f"\n{'='*40}")
    print(f"Overall score: {overall_score:.2f}%")
    print(f"{'='*40}")

    # Save results
    save_results(output_dir, overall_score, per_example, config)

    # MLflow finalize
    finalize_mlflow(mlflow_run, overall_score, output_dir)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
