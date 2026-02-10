PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop retrieval system for the HoVer (Hop-Verify) benchmark, which validates claims by retrieving supporting documents through iterative information-seeking hops. It uses DSPy to orchestrate a three-hop retrieval pipeline that progressively refines queries to find relevant evidence documents.

**Key Modules**:
- **HoverMultiHopPredictPipeline** (hover_pipeline.py): Top-level wrapper that initializes the ColBERTv2 retrieval model and orchestrates the forward pass through the program
- **HoverMultiHopPredict** (hover_program.py): Core multi-hop retrieval logic implementing a 3-hop search strategy with query generation and summarization at each hop
- **hoverBench** (hover_data.py): Dataset loader that filters the HoVer dataset to 3-hop examples and prepares training/test sets
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting documents are found within the top 21 retrieved documents

**Data Flow**:
1. Input claim is passed to HoverMultiHopPredictPipeline
2. Hop 1: Retrieve k=7 documents using the claim, generate summary
3. Hop 2: Create new query from claim + summary_1, retrieve k=7 more docs, generate summary_2
4. Hop 3: Create final query from claim + both summaries, retrieve k=7 final docs
5. Return concatenated list of all 21 retrieved documents (7 per hop)
6. Metric validates if gold supporting facts' document titles are present in retrieved set

**Optimization Target**: The discrete_retrieval_eval metric measures recall - whether all required supporting documents are successfully retrieved within the 21-document budget across the three hops.

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

