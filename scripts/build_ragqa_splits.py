"""Build train/val/test splits for RAGQAArenaTech.

Downloads the source QA file (question + gold ``response``) and writes three
disjoint, deterministically-shuffled splits to the top-level ``data/`` directory:

    data/RAGQAArenaTech_train.json
    data/RAGQAArenaTech_val.json
    data/RAGQAArenaTech_test.json

There is intentionally NO dev split (folded into train) per the CodeEvolver
setup. These files contain GOLD ANSWERS and must be denied to the iteration
architect; the retrieval database (langProBe/RAGQAArenaTech/data/{test_collection.jsonl,index.pt})
is separate and may be read freely.

Usage:
    python scripts/build_ragqa_splits.py            # build if missing
    python scripts/build_ragqa_splits.py --force    # rebuild
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

SOURCE_URL = "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json"

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "ragqa_arena_tech_500.json"

# Split sizes (disjoint). The source file holds ~2064 examples, so these draw
# from a comfortable surplus (300+600+200=1100 << 2064). val is sized at 600 so a
# run can set maxValSetSize up to 600 (e.g. 300) and still have variance headroom
# for a near-ceiling log-odds metric; train is sized for optimization. Because the
# carve order is train->val->test, train stays byte-identical when val grows and the
# new val is a superset of the previous 150-row val. Adjust here to retune.
SIZES = {"train": 300, "val": 600, "test": 200}

# Fixed seed so the partition is reproducible across machines/runs.
SHUFFLE_SEED = 0


def _download_raw() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PATH.exists():
        return
    print(f"Downloading source QA file -> {RAW_PATH}")
    urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)


def build(force: bool = False) -> None:
    _download_raw()

    with open(RAW_PATH) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} source examples (fields: {sorted(raw[0].keys())})")

    total_needed = sum(SIZES.values())
    if total_needed > len(raw):
        raise ValueError(
            f"Requested {total_needed} examples but source only has {len(raw)}"
        )

    # Deterministic shuffle, then carve disjoint contiguous slices.
    indices = list(range(len(raw)))
    random.Random(SHUFFLE_SEED).shuffle(indices)

    cursor = 0
    splits: dict[str, list] = {}
    for name in ("train", "val", "test"):
        n = SIZES[name]
        chosen = indices[cursor : cursor + n]
        cursor += n
        splits[name] = [raw[i] for i in chosen]

    # Sanity: disjoint, no overlap.
    all_idx = indices[:cursor]
    assert len(all_idx) == len(set(all_idx)) == total_needed, "splits overlap!"

    DATA_DIR.mkdir(exist_ok=True)
    for name, examples in splits.items():
        path = DATA_DIR / f"RAGQAArenaTech_{name}.json"
        if path.exists() and not force:
            print(f"SKIP existing {path} ({len(examples)} would be written); use --force")
            continue
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Wrote {path} ({len(examples)} examples)")

    print(
        "\nSplit summary: "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        + " | disjoint, no dev set"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAGQAArenaTech train/val/test splits")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files")
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
