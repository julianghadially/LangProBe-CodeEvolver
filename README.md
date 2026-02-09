# LangProBe-CodeEvolver
This project is a fork of LangProBe.

LangProBe is an AI system benchmark built in DSPY for executing multiple common system types, including the following:
- Multi-hop question-answering - HotpotQA
- Multi-hop fact-checking - HoVer
- RAG question-answering in tech domain - RAGQAArenaTech
- Math question-answering - GSM8k
- Task completion - AppWorld
- Household Task completion - AlfWorld

CodeEvolver is an optimizer that takes an initial program and makes changes to the prompts and the code (including context pipeline, tooling, AI modules, AI module graph, etc.). 

The LangProBe-CodeEvolver benchmark is set up to measure CodeEvolver changes. To produce a useful measurement of impact, CodeEvolver controls for unfair resource changes. For example, the number of hops allowed in the multi hop benchmarks is kept constant. However, in its nature, CodeEvolver is designed to modify the system which inevitably changes resource consumption, including changing the number of AI modules called, and the services used. The controls Ensure that any resource consumption changes are fair.

## LangProBe Usage

### Setup
```bash
conda create -n langprobe python=3.10 -y
conda activate langprobe
pip install -r requirements.txt
```

### How the flow works
Calling evaluation.py kicks off the folowing processes
- langProBe.register_benchmark(".hotpotQA")  # imports hotpotQA/__init__.py
- gets BenchmarkMeta (dataset class, list of programs, metric)
- for each program:
    - EvaluateBench(benchmark, program, metric, lm)
    - evaluate_baseline()
        - dspy.evaluate.Evaluate runs program.forward() on each test example
        - score with answer_exact_match
        - returns score + token/cost stats

### How to Run the Benchmark for CodeEvolver

Each benchmark should be run as follows:
```bash
#make cache directory (modify as needed)
mkdir eval_hover 
export DSPY_CACHEDIR=eval_hover/.dspy_cache

#Hover
python -m langProBe.evaluation --benchmark hover --lm openai/gpt-4.1-mini --dataset_mode test --program HoverMultiHop

#HotpotGEPA
python -m langProBe.evaluation --benchmark langProPlus.hotpotGEPA --lm openai/gpt-4o-mini --dataset_mode test --program HotpotMultiHop
```
Key CLI flags:
  - --benchmark hotpotQA — run a single benchmark (otherwise it runs all benchmarks and programs in aggregate)
  - --program `classname` run a single program (otherwise it runs all programs in the current benchmark?)
  - --benchmark_set nonagent|agent|full — run a category of benchmarks in aggregate
  - --lm openai/gpt-4o-mini — which LLM to use
  - --dataset_mode test|tiny|lite|full — controls dataset size (50/200/500/all)
  - --num_threads 16 — parallelism
  - --use_devset — evaluate on dev set instead of test set


## Quick Usage
```bash
# example with using gpt-4o, with all non-agent datasets
mkdir evaluation_gpt4o
DSPY_CACHEDIR=evaluation_gpt4o/.dspy_cache python -m langProBe.evaluation --benchmark_set=nonagent --file_path=evaluation_gpt4o --lm=openai/gpt-4o
```

## Adding Benchmarks, Programs, Optimizers

Benchmarks and programs are defined by the `BenchmarkMeta` class. You can program definitions to existing `BenchmarkMeta`s or define your own `BenchmarkMeta`s.
Additionally, each `BenchmarkMeta` object also has an `optimizers` field, containing optimizer definitions. You can inspect `optimizers.py` to checkout how to define an optimizer and default optimizers in `DEFAULT_OPTIMIZERS`. Currently, optimizers can only be evaluated with DSPy programs.


## LangProBe Structure

### File structure for each benchmark
```
langProBe/
    bench_name/
        __init__.py
        bench_name_data.py
        bench_name_program.py
        bench_name_utils.py
    ...
```

#### `__init__.py`
This file should define `benchmark: List[BenchmarkMeta]`.

Each `BenchmarkMeta` should have the following fields:
- `benchmark: type[Benchmark]` - the benchmark class
- `programs: List[type[dspy.Module]]` - the programs for this benchmark
- `metric` - the metric for this benchmark

#### `bench_name_data.py`
This file defines the data used for this benchmark. It should download the data, preprocess it, and create a `Benchmark` subclass called `BenchNameBench`. 

#### `bench_name_program.py`
This file defines programs for this benchmark. There are a few requirements for customized programs:
1. `setup_lm(lm:str, api_key:str, api_base:str)` - your program should support setting up lm with `setup_lm` method. We recommend the LiteLLM library.
2. the program should provide a callable interface (through `__call__` method). The arguments will be a dictionary (`**kwargs`) of input keys to their actual values. For example, a question-answering dataset may use the following input `program(question="what is 1 + 1?")`. The answer should be a **DotDict** with respect field. The same example could be `return DotDict({"answer": "2"})`.

#### `bench_name_utils.py`
This file defines utility functions for this benchmark.

