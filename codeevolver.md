PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Program Overview
This is a multi-hop document retrieval system built on DSPy for the HoVer (Hover-nlp) fact verification benchmark. The system performs iterative retrieval across three hops to gather supporting documents for claim verification tasks.

## Key Modules

**HoverMultiHopPredictPipeline** (`hover_pipeline.py`): Top-level pipeline wrapper that initializes ColBERTv2 retrieval model and coordinates the prediction flow. Inherits from both LangProBeDSPyMetaProgram and dspy.Module.

**HoverMultiHopPredict** (`hover_program.py`): Core multi-hop retrieval program implementing a 3-hop strategy. Each hop retrieves k=7 documents, generates summaries, and formulates new queries based on accumulated context. Uses DSPy Predict and Retrieve modules for query generation and document retrieval.

**hover_data.py**: Data loading module that filters HoVer dataset examples to 3-hop cases, reformats them into DSPy examples, and creates train/test splits.

**hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

## Data Flow
1. Input claim flows through HoverMultiHopPredictPipeline.forward()
2. First hop: Direct retrieval on claim, then summarization
3. Second hop: Generate new query from claim+summary1, retrieve, summarize with context
4. Third hop: Generate query from all previous context, retrieve documents
5. Return all retrieved documents (21 total: 7 per hop)
6. Metric evaluates if gold supporting fact titles are subset of retrieved document titles

## Metric
The `discrete_retrieval_eval` metric returns boolean success: whether all gold supporting document titles (normalized) are found within the top 21 retrieved documents. This measures retrieval recall for multi-hop reasoning tasks.

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

