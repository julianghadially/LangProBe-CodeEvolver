PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This program implements a **multi-hop document retrieval system** for fact verification using the HoVer dataset. It performs iterative retrieval across three hops to gather supporting evidence for claim verification.

### Key Modules

1. **HoverMultiHopPredictPipeline** (`hover_pipeline.py`): Top-level wrapper that configures the ColBERTv2 retrieval model and orchestrates the entire pipeline. Sets up the remote ColBERT server connection and invokes the core multi-hop program.

2. **HoverMultiHopPredict** (`hover_program.py`): Core reasoning module implementing the three-hop retrieval strategy. Each hop retrieves k=7 documents, generates summaries, and creates progressively refined queries for the next hop.

3. **hover_utils.py**: Provides the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting documents are found within the retrieved documents (max 21 docs).

4. **hoverBench** (`hover_data.py`): Dataset management class that loads and preprocesses the HoVer dataset, filtering for 3-hop examples from training and validation sets.

### Data Flow

1. Input: A claim requiring verification
2. **Hop 1**: Retrieve k docs using the claim directly, generate summary
3. **Hop 2**: Create refined query from claim + summary_1, retrieve k more docs, generate summary_2
4. **Hop 3**: Create final query from claim + both summaries, retrieve k final docs
5. Output: Concatenated list of all retrieved documents (21 total)
6. Evaluation: Check if gold supporting facts are present in retrieved set

### Metric Optimization

The `discrete_retrieval_eval` metric is a binary success indicator: returns True if all gold supporting document titles (normalized) are found within the top 21 retrieved documents. This optimizes for **recall** of relevant supporting evidence across multiple reasoning hops.

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

