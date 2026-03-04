# LangProBe-CodeEvolver
Copyright © 2026 440 Labs LLC

LangProBe is an AI system benchmark built in DSPY for executing multiple common system types, including the following:
- Multi-hop question-answering - HotpotQA
- Multi-hop fact-checking - HoVer
- RAG question-answering in tech domain - RAGQAArenaTech
- Math question-answering - GSM8k
- Task completion - AppWorld
- Household Task completion - AlfWorld

CodeEvolver will optimize one program at a time, by starting with the initial program and making changes to the prompts and the code (including context pipeline, tooling, AI modules, AI module graph, etc.). 

In changing the system and code, CodeEvolver fundamentally modifies the resource consumption resulting from changing the number of AI modules called, and the services used. However, CodeEvolver does control for unfair resource additions. For example, the number of hops allowed in the multi hop benchmarks is kept constant. See controls by program, below. 

## CodeEvolver
CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric.

This combines several mechanisms:
- **Optimizer algorithm:** GEPA is a reflective language model algorithm that makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer. 
- **Git branching:** A git process manages evolving code across many git worktrees  
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc. 

CodeEvolver and the optimizer lives in its own separate repository. 
CodeEvolver repository: https://github.com/julianghadially/CodeEvolver
CodeEvolver requirements: github repo with module path, metric path, and dataset. No main function required. 

Users connect their code with the CodeEvolver GitHub app, which allows CodeEvolver to add and run code in new branches.

## Programs

### Multihop QA
HotpotQA is designed to contain information from Wikipedia. 
#### What's Allowed
- The program is not required to stay on Wikipedia only.
- The program is allowed to create or remove modules, dynamic prompts, tool calls, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.ChainOfThought, dspy.Predict, etc.)
- There is no limit on the number of search results to display per query
- Available services: Firecrawl and serper.dev. 

#### Constraints:
- Do NOT search more than two times per question. This is a hard requirement.
- Do NOT visit more than one page per query
- Do NOT use the HotpotQA dataset as context. 

### Hover
Hover is designed to retrieve information from 2017 Wikipedia Abstracts (5.9M). 

#### What's Allowed
- The program is not required to stay on Wikipedia only.
- The program is allowed to create or remove modules, dynamic prompts, tool calls, etc.
- The program is allowed to change the module types (e.g., dspy.ReAct for tool calling, dspy.ChainOfThought, dspy.Predict, etc.)
- The program is allowed to add rerankers, provided the final remains the same - 21 total documents
- There is no limit on the number of search results to display per query
- Available services: wikipedia colbert-server (Via dspy.Retrieve), Firecrawl, serper.dev. 

#### Constraints:
- Do NOT search more than three times per question. This is a hard requirement.
- Do NOT return more than 21 documents. This is a hard requirement.
- Do NOT use the hover dataset as context. 


## Experiment
We will be replicating individual LangProBe benchmark programs with CodeEvolver, which provides prompt and architecture optimization.

We use the same LangProbe training, validation, and testing sets.



## simple_eval Pipelines

`simple_eval` provides transparent, per-example evaluation entrypoints for LangProBe programs, with optional MLflow tracing.

### HotpotQA

`simple_eval/evaluate_hover.py` evaluates the Hotpot multi-hop QA pipeline in a transparent evaluation pipeline for HotpotQA with per-example logging and optional MLflow tracing

#### Usage

```bash
# Baseline on test set
python -m simple_eval.evaluate_hotpot --split test

# GEPA-optimized on test set
python -m simple_eval.evaluate_hotpot --split test \
    --program_path gepa_optimize/output_promptonly_gepa/gepa_optimized_program.json
```

Hotpot uses the following components:

- **Module path (program)**: `langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPipeline`
- **Metric paths**:
  - Exact match: `dspy.evaluate.answer_exact_match`
  - Resource-penalty metric: `langProPlus.hotpotGEPA.hotpot_metric_resource.hotpot_accuracy_with_resource_penalty_feedback`
  - LLM judge metric: `langProPlus.hotpotGEPA.hotpot_metric_resource.hotpot_llm_judge_feedback`
- **Dataset**: JSON files in `data/HotpotQABench_<split>.json` (train/dev/val/test).

Results are saved to `simple_eval/results/<label>_<split>_<timestamp>/`.

### Hover

`simple_eval/evaluate_hover.py` evaluates the Hover multi-hop fact-checking retrieval pipeline.

#### Usage

```bash
# Baseline Hover retrieval on validation-as-test split
python -m simple_eval.evaluate_hover --split test
```

Hover uses the following components:

- **Module path (program)**: `langProBe.hover.hover_pipeline.HoverMultiHopPipeline`
- **Metric path (document retrieval)**: `simple_eval.programs.hover.hover_doc_retrieval`
- **Dataset**: HuggingFace `hover-nlp/hover` (train and validation) with 3-hop filtering, as in `langProBe.hover.hover_data.hoverBench`.

Results are saved to `simple_eval/results/<label>_<split>_<timestamp>/`.

## GEPA optimization

`gepa_optimize/run_gepa.py` runs GEPA prompt optimization for Hotpot or Hover. One entrypoint supports multiple programs via `--program`.

- **Hotpot** (default): metric is answer exact match with textual feedback (`ScoreWithFeedback`) for GEPA reflection. Data: `data/HotpotQABench_train.json`, `data/HotpotQABench_val.json`.
- **Hover**: metric is document retrieval (all gold supporting docs in top-21); wrapped as `ScoreWithFeedback` for GEPA. Data: `data/hoverBench_train.json`, `data/hoverBench_val.json`.

### Usage

```bash
# Hotpot (default)
python -m gepa_optimize.run_gepa --program hotpot --seed 7 --auto heavy \
    --lm openai/gpt-4.1-mini --reflection_lm openai/gpt-4.1

# Hover
python -m gepa_optimize.run_gepa --program hover --seed 7 --auto heavy \
    --lm openai/gpt-4.1-mini --reflection_lm openai/gpt-4.1
```

Shared core lives in `gepa_optimize/gepa_core.py`; program-specific loaders, metrics, and preflight live in `gepa_optimize/programs/hotpot.py` and `gepa_optimize/programs/hover.py`.

## Additional Programs
Additional programs are added to `/langProPlus` and their requirements files are mapped in `codeevolver/LangProPlus.md`.