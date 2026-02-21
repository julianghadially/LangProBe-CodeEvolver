PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric

## Architecture Summary

**Purpose**: This program implements a multi-hop question-answering system for the HotpotQA benchmark, which requires reasoning over multiple documents to answer complex factoid questions.

**Key Modules**:
- `HotpotMultiHopPipeline`: Top-level pipeline implementing a 2-hop retrieve-then-extract architecture. Configures ColBERTv2 retrieval model and orchestrates the entire QA process by retrieving passages in two hops, then extracting the answer directly from all passages.
- `ExtractShortAnswer`: DSPy signature for extracting minimal factoid answers directly from passages without summarization
- `HotpotQABench`: Dataset loader that processes the HotpotQA fullwiki dataset into training and test sets

**Data Flow**:
1. Question input triggers first retrieval (Hop 1) using ColBERTv2 to fetch k=10 passages
2. A focused hop 2 query is generated directly from the question and hop 1 passages using chain-of-thought prompting (no summarization)
3. Second retrieval (Hop 2) fetches k=10 additional passages based on the refined query
4. All passages from both hops are concatenated into a single context string
5. The minimal factoid answer is extracted directly from the concatenated passages using dspy.Predict(ExtractShortAnswer)

**Metric**: The system optimizes for `exact_match_metric` (dspy.evaluate.answer_exact_match), which measures whether the generated answer exactly matches the gold answer string from the HotpotQA dataset.

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

