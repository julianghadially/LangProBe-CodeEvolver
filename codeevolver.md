PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Program Purpose
This program implements a multi-hop question answering system for the HotpotQA benchmark, which requires reasoning across multiple documents to answer complex factual questions. It uses a retrieval-augmented generation (RAG) approach with three hops: two retrieval-and-summarization steps followed by answer generation.

## Key Modules

**HotpotMultiHopPredictPipeline** (hotpot_pipeline.py): Top-level pipeline wrapper that configures the ColBERTv2 retrieval model and delegates to the core program.

**HotpotMultiHopPredict** (hotpot_program.py): Core DSPy program implementing the multi-hop reasoning logic. Contains five learned components: retrieve_k (retrieval), create_query_hop2 (query refinement), summarize1/summarize2 (summarization), and generate_answer (answer generation).

**HotpotQABench** (hotpot_data.py): Benchmark dataset loader that fetches HotpotQA fullwiki data from HuggingFace, preprocesses examples with question/answer/supporting titles, and creates shuffled train/test splits.

## Data Flow
1. Initial question → Hop 1 retrieval (k=7 docs) → Summarize passages
2. Question + summary_1 → Generate hop 2 query → Hop 2 retrieval → Summarize with context
3. Question + both summaries → Generate final answer
4. Return prediction with answer field

## Metric
The system optimizes for **exact_match_metric** (dspy.evaluate.answer_exact_match), which measures exact string matching between predicted and gold answers, providing a strict accuracy measure for factual correctness.

## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response

