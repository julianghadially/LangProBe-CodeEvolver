"""
GEPA prompt optimization for LangProBe programs (Hotpot, Hover).

Usage:
    # Hotpot (default)
    python -m gepa_optimize.run_gepa --seed 7 --auto heavy \\
        --lm openai/gpt-4.1-mini --reflection_lm openai/gpt-4.1

    # Hover
    python -m gepa_optimize.run_gepa --program hover --seed 7 --auto heavy \\
        --lm openai/gpt-4.1-mini --reflection_lm openai/gpt-4.1
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import dspy

# Add project root to path so we can import langProPlus / langProBe
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gepa_optimize import gepa_core


def _get_program_module(program_name: str):
    """Resolve program name to its module (load_train_val, load_program, get_metric_for_gepa, check_preflight)."""
    if program_name == "hotpot":
        from gepa_optimize.programs import hotpot # type: ignore[reportMissingImports]
        return hotpot
    if program_name == "hover":
        from gepa_optimize.programs import hover # type: ignore[reportMissingImports]

        return hover
    raise ValueError(f"Unknown program: {program_name}. Choose from: hotpot, hover")


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization on a LangProBe program (Hotpot or Hover)"
    )
    parser.add_argument(
        "--program",
        type=str,
        default="hotpot",
        choices=["hotpot", "hover"],
        help="Program to optimize (default: hotpot)",
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default=None,
        help="Path to saved program JSON (e.g. gepa_optimize/output_.../gepa_optimized_program.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for GEPA",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Student/inference LM",
    )
    parser.add_argument(
        "--reflection_lm",
        type=str,
        default="openai/gpt-4.1",
        help="Reflection LM for GEPA proposals",
    )
    parser.add_argument(
        "--auto",
        type=str,
        default="auto",
        choices=["light", "medium", "heavy"],
        help="GEPA budget: light/medium/heavy",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Parallel eval threads",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=150,
        help="Val set subsample size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-timestamped if not set)",
    )
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = Path(f"gepa_optimize/output_{args.program}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Program: {args.program}")

    # Program-specific wiring
    mod = _get_program_module(args.program)

    # Configure LMs (student LM; reflection_lm is created inside gepa_core)
    student_lm = dspy.LM(args.lm)
    dspy.configure(lm=student_lm, experimental=True)
    print(f"Student LM: {args.lm}")
    print(f"Reflection LM: {args.reflection_lm}")

    # Preflight
    mod.check_preflight()

    # Load dataset
    print(f"\nLoading {args.program} train and val sets...")
    train_set, val_set = mod.load_train_val(PROJECT_ROOT, args.val_size, seed=42)
    print(f"Train set: {len(train_set)} examples")
    print(f"Val set: {len(val_set)} examples (sampled with seed=42)")

    # Load program
    program = mod.load_program(args.program_path)
    metric = mod.get_metric_for_gepa()

    # Run GEPA (compile, save, evaluate on val)
    _, val_score_value = gepa_core.run_gepa_optimization(
        program,
        train_set,
        val_set,
        metric,
        output_dir,
        args,
    )

    # Save results
    gepa_core.save_gepa_results(
        output_dir,
        val_score_value,
        args,
        len(train_set),
        len(val_set),
    )


if __name__ == "__main__":
    main()
