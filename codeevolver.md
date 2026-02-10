PARENT_MODULE_PATH: langProBe.hover.hover_program.HoverMultiHopPredict
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPredict is a multi-hop document retrieval system built on DSPy for fact verification tasks. It performs iterative retrieval and summarization across three hops to gather supporting evidence for claim verification from the HoVer dataset.

**Key Modules**:
- **HoverMultiHopPredict** (hover_program.py): Core retrieval pipeline that orchestrates three-hop iterative search. Each hop retrieves k=7 documents using query generation and summarization to refine subsequent queries.
- **hoverBench** (hover_data.py): Dataset handler that loads and preprocesses the HoVer claim verification dataset, filtering for 3-hop examples from the training and validation sets.
- **hover_utils**: Provides the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting documents are found within the top 21 retrieved documents.
- **LangProBeDSPyMetaProgram** (dspy_program.py): Base class providing DSPy integration and language model configuration.

**Data Flow**:
1. Input claim is used to retrieve initial documents (hop1)
2. Hop1 docs are summarized to generate a refined query for hop2
3. Hop2 docs and previous context create another query for hop3
4. All retrieved documents (21 total: 7×3 hops) are returned as the final prediction
5. Evaluation checks if gold document titles (normalized) are subset of retrieved document titles

**Metric**: `discrete_retrieval_eval` returns binary success (True/False) based on whether all supporting facts from the gold standard are present in the retrieved document set, measuring retrieval recall.

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

