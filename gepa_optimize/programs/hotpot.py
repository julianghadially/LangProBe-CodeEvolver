"""Hotpot-specific dataset, program, metric, and preflight for GEPA optimization."""

from __future__ import annotations

import json
import random
from pathlib import Path

import requests as _requests

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPipeline, COLBERT_URL


def load_train_val(
    project_root: Path,
    val_size: int,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load train and val from HotpotQABench JSON; subsample val with fixed seed 42."""
    train_path = project_root / "data" / "HotpotQABench_train.json"
    val_path = project_root / "data" / "HotpotQABench_val.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Train set not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val set not found: {val_path}")

    with open(train_path) as f:
        train_raw = json.load(f)
    with open(val_path) as f:
        val_raw = json.load(f)

    train_set = [
        dspy.Example(
            question=ex["question"],
            answer=ex["answer"],
            gold_titles=ex.get("gold_titles", []),
        ).with_inputs("question")
        for ex in train_raw
    ]
    val_set = [
        dspy.Example(
            question=ex["question"],
            answer=ex["answer"],
            gold_titles=ex.get("gold_titles", []),
        ).with_inputs("question")
        for ex in val_raw
    ]

    if len(val_set) > val_size:
        val_set = random.Random(seed).sample(val_set, val_size)

    return train_set, val_set


def load_program(program_path: str | None) -> dspy.Module:
    """Create HotpotMultiHopPipeline, optionally loading saved state."""
    program = HotpotMultiHopPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        program.load(str(path))
        print(f"Loaded program state from {path}")
    return program


def hotpot_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric with textual feedback for GEPA optimization.

    Returns score + feedback about answer correctness and gold retrieval titles.
    """
    score = dspy.evaluate.answer_exact_match(gold, pred)

    feedback = f"Gold answer: '{gold.answer}'. Predicted: '{pred.answer}'. "
    if not score:
        feedback += "The answer is INCORRECT. "
    else:
        feedback += "The answer is CORRECT. "

    gold_titles = gold.get("gold_titles", [])
    if gold_titles:
        feedback += f"Supporting documents needed: {gold_titles}."

    return ScoreWithFeedback(score=float(score), feedback=feedback)


def get_metric_for_gepa():
    """Return the Hotpot GEPA metric (ScoreWithFeedback)."""
    return hotpot_metric_with_feedback


def check_preflight() -> None:
    """Verify ColBERT retrieval server is reachable for Hotpot."""
    print(f"Checking ColBERT server at {COLBERT_URL}...")
    try:
        resp = _requests.get(COLBERT_URL, params={"query": "test", "k": 1}, timeout=15)
        resp.raise_for_status()
        print(f"ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: ColBERT server unreachable: {e}")
        print("The pipeline requires ColBERT for retrieval. Proceeding anyway, but expect errors.")
