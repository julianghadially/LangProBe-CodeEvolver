"""Hover-specific dataset, program, metric, and dependency wiring for simple_eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Callable

import dspy

from langProBe.hover.hover_utils import (
    discrete_retrieval_eval,
    MAX_RETRIEVED_DOCS,
)
from langProBe.hover.hover_pipeline import (
    HoverMultiHopPipeline,
    COLBERT_URL as HOVER_COLBERT_URL,
)

import requests as _requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _hover_split_to_filename(split: str) -> str:
    """Map CLI split name to data file suffix."""
    # We support explicit files for train/dev/val/test.
    if split not in {"train", "dev", "val", "test"}:
        raise ValueError(f"Unsupported Hover split: {split}")
    return f"hoverBench_{split}.json"


def load_dataset_hover(
    split: str,
    n: int | None,
    seed: int | None,
) -> List[dspy.Example]:
    """Load Hover examples from preprocessed JSON files in the data/ folder.

    Files:
        data/hoverBench_train.json
        data/hoverBench_dev.json
        data/hoverBench_val.json
        data/hoverBench_test.json
    """
    filename = _hover_split_to_filename(split)
    path = PROJECT_ROOT / "data" / filename
    if not path.exists():
        raise FileNotFoundError(f"Hover dataset not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    examples = [
        dspy.Example(
            claim=ex["claim"],
            supporting_facts=ex["supporting_facts"],
            label=ex["label"],
        ).with_inputs("claim")
        for ex in raw
    ]

    if seed is not None and n is not None:
        rng = __import__("random").Random(seed)
        if n > len(examples):
            print(
                f"WARNING: requested n={n} but only {len(examples)} Hover examples available"
            )
            n = len(examples)
        examples = rng.sample(examples, n)
    elif n is not None:
        examples = examples[:n]

    return examples


def load_program_hover(program_path: str | None) -> dspy.Module:
    """Create HoverMultiHopPipeline, optionally loading saved state.

    Currently, the Hover meta-program does not expose a serialized-state
    interface, so program_path is accepted for API symmetry but ignored.
    """
    program = HoverMultiHopPipeline()
    if program_path:
        # Placeholder for future GEPA-optimized Hover programs.
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        print(
            f"NOTE: program_path={path} is acknowledged but HoverMultiHopPipeline "
            "does not currently support .load(); using baseline pipeline."
        )
    return program


def hover_doc_retrieval(gold, pred, trace=None):
    """Document retrieval metric for Hover.

    Uses the same discrete retrieval evaluation as LangProBe Hover benchmark:
    all gold supporting document keys must appear within the top-K retrieved
    docs (K == MAX_RETRIEVED_DOCS).
    """
    # Reuse the canonical implementation for consistency.
    return float(discrete_retrieval_eval(gold, pred, trace=trace))


def select_metric_hover(metric_name: str | None) -> Tuple[Callable, str]:
    """Return the Hover metric function and its normalized name."""
    # Currently we only expose the document-retrieval metric.
    return hover_doc_retrieval, "hover_doc_retrieval"


def check_dependencies_hover() -> None:
    """Verify ColBERT retrieval server is reachable for Hover."""
    print(f"Checking Hover ColBERT server at {HOVER_COLBERT_URL}...")
    try:
        resp = _requests.get(
            HOVER_COLBERT_URL, params={"query": "test", "k": 1}, timeout=15
        )
        resp.raise_for_status()
        print(f"Hover ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: Hover ColBERT server unreachable: {e}")
        print(
            "The Hover pipeline requires ColBERT for retrieval. "
            "Proceeding anyway, but expect errors."
        )

