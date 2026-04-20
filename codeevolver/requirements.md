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
CodeEvolver requirements: GitHub repo with module path, metric path, and dataset. No standalone `main` entry point is required.

### Program entry contract (DSPy / OpenTelemetry)

CodeEvolver invokes your program as a **callable** that must emit OpenTelemetry traces compatible with DSPy instrumentation.

- **Plain function:** Configure OTEL for DSPy before the function runs. Call `DSPyInstrumentor().instrument()` from `openinference.instrumentation.dspy` at import time (or equivalent), then use `dspy.configure(lm=...)` so the LM is set for that callable’s execution.
- **Class:** The class must implement `__call__(self, ...)`. Perform one-time setup in `__init__` (including `DSPyInstrumentor().instrument()` if not already done at module import, `dspy.LM(...)`, and `dspy.configure(lm=self.lm)`). Each `__call__` should return the same kind of outcome your metric expects—for LangProBe DSPy programs that is typically a `dspy.Prediction` (optionally with extra attributes such as `retrieval_count`), matching `forward` on the corresponding `dspy.Module` pipeline class.

Example in this repo: `langProPlus/hotpotGEPA/hotpot_pipeline.py` defines `HotpotMultiHopProgram`, which mirrors `HotpotMultiHopPipeline.forward`: it returns the program’s `Prediction` with `retrieval_count` set, not a separate dict wrapper.

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



## simple_eval Pipeline

`simple_eval/evaluate.py` is a transparent, single-file evaluation pipeline for HotpotQA with per-example logging and optional MLflow tracing.

### Diagnostic Workflow

```bash
# Baseline on test set
python -m simple_eval.evaluate --split test

# GEPA-optimized on test set
python -m simple_eval.evaluate --split test \
    --program_path gepa_optimize/output_promptonly_gepa/gepa_optimized_program.json
```

Results are saved to `simple_eval/results/<label>_<split>_<timestamp>/`.

## Additional Programs
Additional programs are added to /langProPlus and their requirements files are mapped in codeevolver/LangProPlus.md