PARENT_MODULE_PATH: langProBe.hover.hover_program.HoverMultiHopPredict
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPredict is a multi-hop fact-checking system built on DSPy that verifies claims by iteratively retrieving and summarizing supporting documents across three hops. The system is designed to find all supporting facts needed to verify claims in the HoVer dataset.

**Key Modules**:
- **HoverMultiHopPredict** (hover_program.py): The main DSPy module implementing a 3-hop retrieval pipeline. It orchestrates query generation, document retrieval (k=7 per hop), and summarization at each hop using specialized DSPy Predict modules (create_query_hop2, create_query_hop3, summarize1, summarize2).
- **hoverBench** (hover_data.py): Data loader that loads the HoVer dataset from Hugging Face, filters examples to 3-hop instances only, and formats them as DSPy examples with the claim as input.
- **discrete_retrieval_eval** (hover_utils.py): The evaluation metric that checks if all gold supporting fact document titles are present in the retrieved documents (up to 21 max). Returns True if gold titles are a subset of found titles.

**Data Flow**: 
1. Input claim → HOP 1: retrieve k docs, summarize → summary_1
2. Generate hop2 query from claim + summary_1 → HOP 2: retrieve k docs, summarize with context → summary_2  
3. Generate hop3 query from claim + summary_1 + summary_2 → HOP 3: retrieve k docs
4. Return concatenated retrieved_docs from all 3 hops (21 total)
5. Metric evaluates if all supporting fact titles are found in the 21 retrieved documents

**Metric Optimized**: The discrete_retrieval_eval metric measures retrieval accuracy - whether all ground truth supporting document titles are successfully retrieved within the 21-document budget (7 per hop × 3 hops).

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

