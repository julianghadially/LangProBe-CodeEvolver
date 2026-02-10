PARENT_MODULE_PATH: langProBe.hover.hover_program.HoverMultiHopPredict
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop fact-checking system for the HoVer benchmark that retrieves supporting evidence documents for claims requiring up to 3 hops of reasoning. Built using DSPy, it performs iterative retrieval-summarization cycles to gather relevant evidence.

**Key Modules**:
- **HoverMultiHopPredict** (hover_program.py): The main pipeline orchestrating 3-hop retrieval. Each hop: (1) generates a search query based on the claim and previous summaries, (2) retrieves k=7 documents, (3) summarizes findings. Uses DSPy modules: `Predict` for query generation and summarization, `Retrieve(k=7)` for document retrieval.
- **hoverBench** (hover_data.py): Dataset loader that fetches HoVer dataset, filters for 3-hop examples (checking `count_unique_docs`), and creates train/dev/val/test splits as DSPy Examples with "claim" as input.
- **discrete_retrieval_eval** (hover_utils.py): Metric function that checks if all gold supporting fact document titles are present in the top 21 retrieved documents (union of all 3 hops). Returns binary success/failure.

**Data Flow**: 
claim → HOP1: retrieve(claim) + summarize → HOP2: generate_query(claim, summary_1) → retrieve + summarize → HOP3: generate_query(claim, summary_1, summary_2) → retrieve → return concatenated documents from all hops → metric validates presence of gold documents

**Optimization Target**: Maximize the discrete retrieval success rate (binary: all required documents found in top 21 retrieved).

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

