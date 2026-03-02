"""
GEPA prompt optimization for HotpotMultiHopPipeline.

Usage:
    python gepa_optimize/run_gepa.py \
        --seed 7 --auto heavy \
        --lm openai/gpt-4.1-mini \
        --reflection_lm openai/gpt-4.1
"""

import argparse
import inspect
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import requests as _requests

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

# Add project root to path so we can import langProPlus
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPipeline, COLBERT_URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "HotpotQABench_train.json"
VAL_PATH = PROJECT_ROOT / "data" / "HotpotQABench_val.json"


def load_examples_from_json(path):
    """Load dspy.Example list from a saved JSON split file."""
    with open(path) as f:
        raw = json.load(f)
    return [
        dspy.Example(question=ex["question"], answer=ex["answer"],
                     gold_titles=ex.get("gold_titles", [])).with_inputs("question")
        for ex in raw
    ]


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


def load_program(program_path=None):
    """Create a HotpotMultiHopPipeline, optionally loading saved state."""
    program = HotpotMultiHopPipeline()
    if program_path:
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program path not found: {path}")
        program.load(str(path))
        print(f"Loaded program state from {path}")
    return program


def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization on HotpotMultiHopPipeline")
    parser.add_argument("--program_path", type=str, default=None,
                        help="Path to saved program JSON (e.g. codeevolver/results/optimized_program.json)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for val set subsampling and GEPA")
    parser.add_argument("--lm", type=str, default="openai/gpt-4.1-mini",
                        help="Student/inference LM")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-4.1",
                        help="Reflection LM for GEPA proposals")
    parser.add_argument("--auto", type=str, default="auto",
                        choices=["light", "medium", "heavy"],
                        help="GEPA budget: light/medium/heavy")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="Parallel eval threads")
    parser.add_argument("--val_size", type=int, default=150,
                        help="Val set subsample size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-timestamped if not set)")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = Path(f"gepa_optimize/output_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Disable SQLite disk cache to avoid contention with multiple threads.
    # In-memory cache is kept (thread-safe LRU).
    dspy.cache.enable_disk_cache = False

    # Configure LMs
    student_lm = dspy.LM(args.lm)
    reflection_lm = dspy.LM(args.reflection_lm, temperature=1.0, max_tokens=16000)
    dspy.configure(lm=student_lm, experimental=True)
    print(f"Student LM: {args.lm}")
    print(f"Reflection LM: {args.reflection_lm}")

    # Preflight: verify ColBERT server is reachable
    print(f"Checking ColBERT server at {COLBERT_URL}...")
    try:
        resp = _requests.get(COLBERT_URL, params={"query": "test", "k": 1}, timeout=15)
        resp.raise_for_status()
        print(f"ColBERT server OK (status {resp.status_code})")
    except Exception as e:
        print(f"WARNING: ColBERT server unreachable: {e}")
        print("The pipeline requires ColBERT for retrieval. Proceeding anyway, but expect errors.")

    # Load dataset from cached JSON splits
    print(f"Loading train set from {TRAIN_PATH}...")
    train_set = load_examples_from_json(TRAIN_PATH)
    print(f"Loading val set from {VAL_PATH}...")
    val_set = load_examples_from_json(VAL_PATH)

    # Subsample val set with seed=42
    if len(val_set) > args.val_size:
        val_set = random.Random(42).sample(val_set, args.val_size)
    print(f"Train set: {len(train_set)} examples")
    print(f"Val set: {len(val_set)} examples (sampled with seed=42)")

    # Load program
    program = load_program(args.program_path)

    # Configure and run GEPA
    print(f"\nStarting GEPA optimization (auto={args.auto}, seed={args.seed})...")
    gepa = dspy.GEPA(
        metric=hotpot_metric_with_feedback,
        auto=args.auto,
        reflection_lm=reflection_lm,
        num_threads=args.num_threads,
        seed=args.seed,
        log_dir=str(output_dir / "gepa_logs"),
        track_stats=True,
    )

    optimized_program = gepa.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    # Save optimized program
    # Workaround: DSPy's Retrieve.dump_state() doesn't accept the json_mode kwarg
    # that BaseModule.dump_state() passes. Patch it before saving.
    from dspy.retrievers.retrieve import Retrieve as _Retrieve
    _orig_dump = _Retrieve.dump_state
    if "json_mode" not in inspect.signature(_orig_dump).parameters:
        _Retrieve.dump_state = lambda self, **kwargs: _orig_dump(self)

    save_path = output_dir / "gepa_optimized_program.json"
    optimized_program.save(str(save_path))

    _Retrieve.dump_state = _orig_dump  # restore
    print(f"\nSaved optimized program to {save_path}")

    # Evaluate on val set
    print("\nEvaluating optimized program on val set...")
    evaluator = Evaluate(
        devset=val_set,
        metric=hotpot_metric_with_feedback,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=5000,
    )
    val_score = evaluator(optimized_program)
    val_score_value = val_score.score if hasattr(val_score, "score") else val_score
    print(f"Final val score: {val_score_value}")

    # Save results
    results = {
        "val_score": val_score_value,
        "args": vars(args),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    results_txt_path = output_dir / "results.txt"
    with open(results_txt_path, "w") as f:
        f.write(f"GEPA Optimization Results\n")
        f.write(f"========================\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Val score: {val_score_value}\n")
        f.write(f"Auto budget: {args.auto}\n")
        f.write(f"Student LM: {args.lm}\n")
        f.write(f"Reflection LM: {args.reflection_lm}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Train size: {len(train_set)}\n")
        f.write(f"Val size: {len(val_set)}\n")
        if args.program_path:
            f.write(f"Starting program: {args.program_path}\n")

    print(f"Saved results to {results_path} and {results_txt_path}")


if __name__ == "__main__":
    main()
