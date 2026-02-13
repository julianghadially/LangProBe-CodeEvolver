```
PARENT_MODULE_PATH: langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPredictPipeline
METRIC_MODULE_PATH: langProPlus.hotpotGEPA.__init__.exact_match_metric
```

## Architecture Summary

**Purpose**: This is a multi-hop question answering system built on DSPy that answers complex questions from the HotpotQA dataset by performing iterative document retrieval and summarization across multiple reasoning steps.

**Key Modules**:
- `HotpotMultiHopPredictPipeline`: Top-level pipeline wrapper that configures the ColBERTv2 retrieval model and orchestrates the entire question-answering flow
- `HotpotMultiHopPredict`: Core program implementing a 3-hop reasoning strategy using DSPy Predict modules for query generation, summarization, and answer generation
- `HotpotQABench`: Benchmark loader that loads and prepares the HotpotQA fullwiki dataset with train/validation splits

**Data Flow**:
1. Input question enters the pipeline wrapper which sets up ColBERTv2 retrieval context
2. **Hop 1**: Retrieve top-k documents for original question, generate summary_1
3. **Hop 2**: Create refined query from question + summary_1, retrieve more documents, generate summary_2
4. **Hop 3**: Generate final answer from question + summary_1 + summary_2 using the GenerateAnswer signature
5. Return answer prediction

**Optimization Metric**: `exact_match_metric` (dspy.evaluate.answer_exact_match) - measures whether the predicted answer exactly matches the ground truth answer string from HotpotQA validation set. The system is optimized to maximize exact string matching accuracy on multi-hop reasoning questions.

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

