"""
Load HotpotQA dataset from HuggingFace and create train/val/test splits.

All splits are sampled from the HuggingFace HotpotQA *train* split (NOT validation),
because the HF validation split is substantially different from HF train.

The test set is created by sampling 2x the desired size, filtering out any rows
already in train or val, then taking the first N rows from the filtered result.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _hf_row_to_dict(x: dict) -> dict:
    """Convert a raw HuggingFace HotpotQA row to our standard format."""
    return {
        "question": x["question"],
        "answer": x["answer"],
        "gold_titles": list(set(x["supporting_facts"]["title"])),
    }


def _load_split(filename: str) -> list[dict] | None:
    """Load an existing split from JSON, or return None if it doesn't exist."""
    path = DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_split(data: list[dict], filename: str) -> None:
    """Save a split to JSON."""
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {path} ({len(data)} examples)")


def _make_question_key(row: dict) -> str:
    """Create a unique key for deduplication based on question text."""
    return row["question"].strip()


def load_and_split(
    train_size: int = 150,
    val_size: int = 300,
    test_size: int = 300,
    seed: int = 0,
):
    """
    Load HotpotQA from HuggingFace and create train/val/test splits.

    - If train already exists on disk, skip it.
    - If val already exists on disk, skip it.
    - If test already exists on disk, skip it.
    - Otherwise, test is created by sampling 2x rows from HF train,
      filtering out any overlap with train/val, then taking the first
      `test_size` rows.

    All splits come from HuggingFace hotpot_qa "fullwiki" train split.
    Files are saved to data/ as HotpotQABench_{train,val,test}.json.
    """

    DATA_DIR.mkdir(exist_ok=True)

    # --- Check existing splits ---
    existing_train = _load_split("HotpotQABench_train.json")
    existing_val = _load_split("HotpotQABench_val.json")
    existing_test = _load_split("HotpotQABench_test.json")

    need_hf = (existing_train is None) or (existing_val is None) or (existing_test is None)

    if need_hf:
        print("Loading HotpotQA from HuggingFace (train split)...")
        raw = load_dataset("hotpot_qa", "fullwiki", split="train")
        all_rows = [_hf_row_to_dict(x) for x in raw]

        rng = random.Random(seed)
        rng.shuffle(all_rows)
        print(f"Loaded {len(all_rows)} rows from HF train split")
    else:
        all_rows = None

    # --- Train split ---
    if existing_train is not None:
        train_data = existing_train
        print(f"Train set already exists ({len(train_data)} rows), skipping.")
    else:
        train_data = all_rows[:train_size]
        _save_split(train_data, "HotpotQABench_train.json")

    # --- Val split ---
    if existing_val is not None:
        val_data = existing_val
        print(f"Val set already exists ({len(val_data)} rows), skipping.")
    else:
        val_data = all_rows[train_size : train_size + val_size]
        _save_split(val_data, "HotpotQABench_val.json")

    # --- Test split ---
    if existing_test is not None:
        test_data = existing_test
        print(f"Test set already exists ({len(test_data)} rows), skipping.")
    else:
        # Build a set of questions already used in train and val for deduplication
        used_questions = set()
        for row in train_data:
            used_questions.add(_make_question_key(row))
        for row in val_data:
            used_questions.add(_make_question_key(row))

        # Sample 2x the desired test size from the HF train data,
        # starting after the train+val region to minimize overlap
        test_sample_start = train_size + val_size
        test_sample_size = test_size * 2

        # Use a separate RNG for test sampling so it's independent
        test_rng = random.Random(seed + 42)
        candidate_pool = all_rows[test_sample_start:]
        test_candidates = test_rng.sample(
            candidate_pool, min(test_sample_size, len(candidate_pool))
        )

        # Filter out any rows whose question appears in train or val
        filtered = [
            row for row in test_candidates
            if _make_question_key(row) not in used_questions
        ]

        test_data = filtered[:test_size]

        if len(test_data) < test_size:
            print(
                f"WARNING: Only got {len(test_data)} unique test rows "
                f"(requested {test_size}). Consider increasing the sample multiplier."
            )

        _save_split(test_data, "HotpotQABench_test.json")

    # --- Summary ---
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_data)} rows")
    print(f"  Val:   {len(val_data)} rows")
    print(f"  Test:  {len(test_data)} rows")

    return train_data, val_data, test_data


if __name__ == "__main__":
    load_and_split()
