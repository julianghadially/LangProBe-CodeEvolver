# LangProBe Benchmark & Evaluation Process
AI-Generated

## Overview

LangProBe evaluates AI system programs by running them against standardized datasets and scoring their outputs with a metric function. The entry point is `evaluation.py`, which orchestrates benchmark registration, program execution, and result collection.

### Flow: `evaluate_all()`

1. **Register benchmarks**: Calls `register_all_benchmarks(benchmarks)` which dynamically imports each benchmark module (e.g., `langProBe.hover`) via `importlib.import_module()`. Each module must export a `benchmark` list of `BenchmarkMeta` objects.

2. **Iterate benchmarks**: For each `BenchmarkMeta`, calls `evaluate()`.

3. **Aggregate results**: After all benchmarks complete, reads all `.txt` result files from the output directory via `analysis.read_evaluation_results()`, compiles them into a pandas DataFrame, and writes `evaluation_results.csv`.

### Flow: `evaluate()` (per benchmark)

1. **Instantiate the benchmark**: Calls `benchmark_meta.benchmark(dataset_mode=...)` to create a `Benchmark` instance, which downloads/loads the dataset and creates train/dev/val/test splits.

2. **Iterate programs**: For each program in `benchmark_meta.program`:
   - Filters by `program_class` and `program_name_filter` if specified.
   - In `missing_mode`, skips experiments that already have result files.

3. **Create `EvaluateBench`**: Instantiates with the benchmark, program, metric, LM, and optimizers. The program's `setup_lm()` is called to configure it with the specified language model.

4. **Run `evaluate_bench.evaluate()`**: Executes the evaluation (see EvaluateBench section below).

5. **Write results**: For each `EvaluationResult`, writes a `.txt` file named `{benchmark}_{program}_{optimizer}.txt` containing score, cost, and token counts. If an optimizer was used, also saves the optimized program as a `.json` file.

## Benchmark Class (`benchmark.py`)

### `Benchmark` (abstract base class)

Handles dataset loading and splitting:

1. **`init_dataset()`** (abstract): Subclasses implement this to download and preprocess data. Must set `self.dataset` and `self.test_set`, where each element is a `dspy.Example`.

2. **Splitting**: If train/dev/val splits aren't pre-defined, `create_splits()` divides the dataset 40/40/20 (dev/val/train).

3. **Trimming**: Splits are capped at fixed sizes regardless of dataset mode:
   - Train: 150 examples
   - Dev: 300 examples
   - Val: 300 examples
   - Test: varies by `dataset_mode` (50/200/500/all)

4. **Saving**: Splits are serialized to `data/{BenchName}_{split}.json` for external use (e.g., by CodeEvolver).

### `BenchmarkMeta` (dataclass)

Bundles everything needed to run a benchmark:

- `benchmark`: The `Benchmark` subclass (dataset provider)
- `program`: List of program instances to evaluate
- `metric`: Scoring function (e.g., `dspy.evaluate.answer_exact_match`)
- `dataset_mode`: Default dataset size (`"lite"`)
- `optimizers`: List of `OptimizerConfig` objects (defaults to `DEFAULT_OPTIMIZERS`)
- `num_threads`: Optional thread cap
- `name`: Optional display name

### `EvaluateBench`

The core evaluation executor:

1. **Initialization**: Takes a benchmark, program, metric, LM, and optional optimizers. Creates a `dspy.evaluate.Evaluate` instance configured with the test set (or dev set if `--use_devset`).

2. **`evaluate()`**: Dispatches based on program type and configured features:
   - **Baseline** (`evaluate_baseline()`): Runs the program's `forward()` on every test example via `dspy.evaluate.Evaluate`, scores with the metric, and records score + token/cost stats from the LM history.
   - **Optimizer** (`evaluate_optimizers()`): For DSPy programs only. For each optimizer, compiles an optimized version of the program using the train set (and optionally val set), then evaluates the optimized program on the test set. Records both optimization cost and evaluation cost separately.
   - **Assertion** (`evaluate_assertion()`): Activates DSPy assertions on the program before evaluating (currently minimal implementation).

   Non-DSPy programs only get baseline evaluation.

3. **Cost tracking**: After each evaluation, `calculate_stats()` sums up cost, input tokens, and output tokens from the LM's call history.

### `EvaluationResult` (dataclass)

Each evaluation produces a result containing:
- `benchmark`, `program`: Identifiers
- `score`: The metric score (e.g., exact match percentage)
- `cost`, `input_tokens`, `output_tokens`: LLM usage during evaluation
- `optimizer`: Name of optimizer used (or None for baseline)
- `optimizer_cost`, `optimizer_input_tokens`, `optimizer_output_tokens`: LLM usage during optimization

## Benchmark Registration (`register_benchmark.py`)

Benchmarks are registered by module path (e.g., `".hover"` imports `langProBe.hover`). Each benchmark module's `__init__.py` must export a `benchmark` variable: a list of `BenchmarkMeta` objects.

Example from `hotpotQA/__init__.py`:
```python
benchmark = [
    BenchmarkMeta(
        HotpotQABench,           # Benchmark subclass
        [HotPotQAPredict, ...],  # List of programs
        dspy.evaluate.answer_exact_match,  # Metric
    )
]
```

For nested benchmarks (e.g., `langProPlus.hotpotGEPA`), the full dotted path is used as the `--benchmark` argument, and the module is imported directly rather than relative to `langProBe`.

## Result Analysis (`analysis.py`)

`read_evaluation_results()` scans the output directory for `.txt` result files, parses filenames (`{benchmark}_{program}_{optimizer}.txt`) and file contents (CSV with score/cost/tokens), and compiles everything into a pandas DataFrame. Program names are canonicalized via a mapping (e.g., `"ChainOfThought"` -> `"CoT"`).

## Result File Format

Each evaluation writes a file `{benchmark}_{program}_{optimizer}.txt`:
```
score,cost,input_tokens,output_tokens[,optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens]
<values>
```

The optimizer columns only appear when an optimizer was used. The final aggregated CSV (`evaluation_results.csv`) combines all result files with the model name appended.
