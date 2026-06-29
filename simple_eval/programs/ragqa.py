"""RAGQAArenaTech-specific dataset, program, metric, and dependency wiring for simple_eval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

import dspy

from langProBe.RAGQAArenaTech.RAGQAArenaTech_pipeline import RAGQAArenaTechPipeline
from langProBe.RAGQAArenaTech.metric import ragqa_semantic_f1


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ragqa_split_to_filename(split: str) -> str:
    """Map CLI split name to data file. There is no dev split (folded into train)."""
    if split in {"dev", "val"}:
        # 'dev' is accepted as an alias for 'val' since RAGQAArenaTech has no dev set.
        split = "val"
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported RAGQAArenaTech split: {split}")
    return f"RAGQAArenaTech_{split}.json"


def load_dataset_ragqa(
    split: str,
    n: int | None,
    seed: int | None,
) -> List[dspy.Example]:
    """Load RAGQAArenaTech examples from the pre-generated split files in data/.

    Files (built by scripts/build_ragqa_splits.py):
        data/RAGQAArenaTech_train.json
        data/RAGQAArenaTech_val.json
        data/RAGQAArenaTech_test.json

    Each record has 'question' (input) and 'response' (gold answer).
    """
    filename = _ragqa_split_to_filename(split)
    path = PROJECT_ROOT / "data" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"RAGQAArenaTech split not found: {path}. "
            "Run: python scripts/build_ragqa_splits.py"
        )

    with open(path) as f:
        raw = json.load(f)

    examples = [
        dspy.Example(question=ex["question"], response=ex["response"]).with_inputs(
            "question"
        )
        for ex in raw
    ]

    if seed is not None and n is not None:
        rng = __import__("random").Random(seed)
        if n > len(examples):
            print(
                f"WARNING: requested n={n} but only {len(examples)} RAGQAArenaTech examples available"
            )
            n = len(examples)
        examples = rng.sample(examples, n)
    elif n is not None:
        examples = examples[:n]

    return examples


def load_program_ragqa(program_path: str | None) -> dspy.Module:
    """Create the RAGQAArenaTechPipeline (wrapping SimplifiedBaleen by default).

    program_path is accepted for API symmetry with other simple_eval programs but
    is a no-op placeholder: the RAGQAArenaTech meta-program does not currently
    expose a serialized-state (.load()) interface.
    """
    program = RAGQAArenaTechPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        print(
            f"NOTE: program_path={path} is acknowledged but RAGQAArenaTechPipeline "
            "does not currently support .load(); using baseline pipeline."
        )
    return program


def select_metric_ragqa(metric_name: str | None) -> Tuple[Callable, str]:
    """Return the RAGQAArenaTech metric and its normalized name.

    Uses the canonical scorer in langProBe/RAGQAArenaTech/metric.py: SemanticF1
    (key-idea recall/precision of the generated response vs the gold response),
    judged by the metric module's JUDGE_LM. The GEPA/CodeEvolver feedback variant
    (ragqa_semantic_f1_feedback) lives alongside it in the same module.
    """
    return ragqa_semantic_f1, "semantic_f1"


def check_dependencies_ragqa() -> None:
    """Preflight checks for RAGQAArenaTech.

    Unlike Hover/Hotpot there is no ColBERT server. Retrieval uses OpenAI
    embeddings (text-embedding-3-small) and the SemanticF1 judge uses gpt-4o,
    so an OpenAI key is required. The corpus + embedding index download into
    langProBe/RAGQAArenaTech/data/ on first program instantiation.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "WARNING: OPENAI_API_KEY is not set. RAGQAArenaTech needs it for both "
            "embedding retrieval (text-embedding-3-small) and the SemanticF1 judge "
            "(gpt-4o). Proceeding anyway, but expect errors."
        )
    else:
        print("OPENAI_API_KEY detected.")
    print(
        "Note: first run downloads the LoTTE 'technology' corpus + index.pt into "
        "langProBe/RAGQAArenaTech/data/ (this is the retrieval database, not gold labels)."
    )
