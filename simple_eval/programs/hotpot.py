"""Hotpot-specific dataset, program, metric, and dependency wiring for simple_eval."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple, Callable

import requests as _requests

import dspy
from dspy.evaluate import answer_exact_match

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_dataset_hotpot(
    split: str,
    n: int | None,
    seed: int | None,
) -> list[dspy.Example]:
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


def load_program_hotpot(program_path: str | None) -> dspy.Module:
    """Create HotpotMultiHopPipeline, optionally loading saved state."""
    from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPipeline

    program = HotpotMultiHopPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        program.load(str(path))
        print(f"Loaded Hotpot program state from {path}")
    return program


def check_dependencies_hotpot() -> None:
    """Verify ColBERT retrieval server is reachable for Hotpot."""
    from langProPlus.hotpotGEPA.hotpot_pipeline import COLBERT_URL

    print(f"Checking ColBERT server at {COLBERT_URL}...")
    try:
        resp = _requests.get(COLBERT_URL, params={"query": "test", "k": 1}, timeout=15)
        resp.raise_for_status()
        print(f"ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: ColBERT server unreachable: {e}")
        print("The Hotpot pipeline requires ColBERT for retrieval. Proceeding anyway, but expect errors.")


def select_metric_hotpot(metric_name: str | None) -> Tuple[Callable, str]:
    """Return the Hotpot metric function and its normalized name."""
    from langProPlus.hotpotGEPA.hotpot_metric_resource import (
        hotpot_accuracy_with_resource_penalty_feedback,
        hotpot_llm_judge_feedback,
    )

    if metric_name == "resource_penalty":
        return hotpot_accuracy_with_resource_penalty_feedback, "accuracy_with_resource_penalty"
    if metric_name == "llm_judge":
        return hotpot_llm_judge_feedback, "llm_judge"

    # Default to standard exact match.
    return answer_exact_match, "answer_exact_match"

