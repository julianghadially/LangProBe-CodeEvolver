"""
Core evaluation utilities shared across simple_eval entrypoints.

This module is intentionally program-agnostic. Program-specific scripts
are responsible for:
- Loading their dataset into a list[dspy.Example]
- Instantiating a dspy.Module program
- Choosing the metric function
- Performing any external dependency preflight checks
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import dspy
from dspy.evaluate import Evaluate, answer_exact_match


def _extract_score(score: Any) -> float:
    """Extract a float from a metric return value (handles ScoreWithFeedback)."""
    if hasattr(score, "score"):
        return float(score.score)
    return float(score)


def run_evaluation(
    program: dspy.Module,
    dataset: Iterable[dspy.Example],
    num_threads: int,
    metric: Callable[..., Any] | None = None,
) -> Tuple[float, list[dict]]:
    """Run dspy.evaluate.Evaluate and extract per-example results.

    Returns (overall_score, per_example_list) where each entry in
    per_example_list is a dict with at least:
        - idx
        - question/claim (or generic 'input' if not present)
        - gold_answer / label (if present)
        - predicted_answer (if present)
        - score

    Program-specific entrypoints may post-process per-example rows further
    if they need additional fields.
    """
    if metric is None:
        metric = answer_exact_match

    evaluator = Evaluate(
        devset=list(dataset),
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=5000,
    )
    eval_result = evaluator(program)

    overall_score = getattr(eval_result, "score", 0.0)
    if hasattr(overall_score, "score"):
        overall_score = overall_score.score

    results_list = getattr(eval_result, "results", [])

    per_example: list[dict] = []
    for idx, (example, prediction, score) in enumerate(results_list):
        # Try to be generic about input / answer field names.
        input_text = getattr(example, "question", None) or getattr(
            example, "claim", None
        )
        if input_text is None:
            # Fallback to a generic representation.
            input_text = str(
                getattr(example, "question", getattr(example, "claim", example))
            )

        gold_answer = getattr(example, "answer", None)
        label = getattr(example, "label", None)

        row: dict[str, Any] = {
            "idx": idx,
            "input": input_text,
            "gold_answer": gold_answer,
            "label": label,
            "predicted_answer": getattr(prediction, "answer", str(prediction)),
            "score": _extract_score(score),
        }

        # Common optional attrs used by some programs.
        retrieval_count = getattr(prediction, "retrieval_count", None)
        if retrieval_count is not None:
            row["retrieval_count"] = int(retrieval_count)

        gold_titles = getattr(example, "gold_titles", None)
        if gold_titles is not None:
            row["gold_titles"] = json.dumps(gold_titles)

        per_example.append(row)

    return float(overall_score), per_example


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
    if per_example:
        # Use keys from first row; order common fields nicely if present.
        fieldnames = list(per_example[0].keys())
        preferred_order = [
            "idx",
            "input",
            "question",
            "claim",
            "gold_answer",
            "label",
            "predicted_answer",
            "score",
            "gold_titles",
            "retrieval_count",
        ]
        ordered = [f for f in preferred_order if f in fieldnames] + [
            f for f in fieldnames if f not in preferred_order
        ]
        fieldnames = ordered
    else:
        fieldnames = ["idx", "input", "gold_answer", "predicted_answer", "score"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_example)
    print(f"Saved per-example results to {csv_path}")

    # Summary JSON
    correct = sum(1 for row in per_example if float(row.get("score", 0.0)) > 0)
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


def setup_mlflow(
    experiment_name: str,
    config: dict,
    tracking_uri: str = "http://127.0.0.1:5000",
):
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


def finalize_mlflow(run, overall_score: float, output_dir: Path) -> None:
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
    except Exception as e:  # pragma: no cover - best-effort logging
        print(f"WARNING: MLflow finalization failed: {e}")


def build_output_dir(
    label: str,
    split: str,
    explicit_output_dir: str | None = None,
) -> Path:
    """Helper to construct or normalize an output directory path."""
    if explicit_output_dir:
        return Path(explicit_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return Path(f"simple_eval/results/{label}_{split}_{timestamp}")

