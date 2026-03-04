"""Hover-specific dataset, program, metric, and preflight for GEPA optimization."""

from __future__ import annotations

import json
import random
from pathlib import Path

import requests as _requests

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from langProBe.hover.hover_utils import discrete_retrieval_eval
from langProBe.hover.hover_pipeline import HoverMultiHopPipeline, COLBERT_URL


def load_train_val(
    project_root: Path,
    val_size: int,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load train and val from hoverBench JSON; subsample val with fixed seed 42."""
    train_path = project_root / "data" / "hoverBench_train.json"
    val_path = project_root / "data" / "hoverBench_val.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Hover train set not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Hover val set not found: {val_path}")

    with open(train_path) as f:
        train_raw = json.load(f)
    with open(val_path) as f:
        val_raw = json.load(f)

    train_set = [
        dspy.Example(
            claim=ex["claim"],
            supporting_facts=ex["supporting_facts"],
            label=ex["label"],
        ).with_inputs("claim")
        for ex in train_raw
    ]
    val_set = [
        dspy.Example(
            claim=ex["claim"],
            supporting_facts=ex["supporting_facts"],
            label=ex["label"],
        ).with_inputs("claim")
        for ex in val_raw
    ]

    if len(val_set) > val_size:
        val_set = random.Random(seed).sample(val_set, val_size)

    return train_set, val_set


def load_program(program_path: str | None) -> dspy.Module:
    """Create HoverMultiHopPipeline. Hover pipeline does not support .load() yet; program_path is ignored."""
    program = HoverMultiHopPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        print(
            "NOTE: program_path is set but HoverMultiHopPipeline does not currently support .load(); using baseline pipeline."
        )
    return program


def hover_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Document retrieval metric with textual feedback for GEPA.

    Uses discrete_retrieval_eval: all gold supporting doc keys must appear in top-21 retrieved.
    Returns ScoreWithFeedback so GEPA's reflection LM gets a scalar and text.
    """
    score_bool = discrete_retrieval_eval(gold, pred, trace=trace)
    score = float(score_bool)

    gold_keys = [doc["key"] for doc in gold.supporting_facts]
    if score_bool:
        feedback = "Retrieval CORRECT: all supporting docs in top-21."
    else:
        feedback = f"Retrieval INCORRECT: gold supporting docs needed: {gold_keys}."

    return ScoreWithFeedback(score=score, feedback=feedback)


def get_metric_for_gepa():
    """Return the Hover GEPA metric (ScoreWithFeedback)."""
    return hover_metric_with_feedback


def check_preflight() -> None:
    """Verify ColBERT retrieval server is reachable for Hover."""
    print(f"Checking Hover ColBERT server at {COLBERT_URL}...")
    try:
        resp = _requests.get(COLBERT_URL, params={"query": "test", "k": 1}, timeout=15)
        resp.raise_for_status()
        print(f"Hover ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: ColBERT server unreachable: {e}")
        print(
            "The Hover pipeline requires ColBERT for retrieval. Proceeding anyway, but expect errors."
        )
