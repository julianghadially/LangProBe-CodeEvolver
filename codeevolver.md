PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop document retrieval system for fact-checking claims using the HOVER dataset. It retrieves relevant supporting documents through iterative retrieval and summarization steps.

**Key Modules**:
- **HoverMultiHopPipeline**: Entry point that configures the ColBERTv2 retriever and orchestrates the program execution
- **HoverMultiHop**: Core retrieval logic implementing a 3-hop retrieval strategy with progressive query refinement
- **hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- **hover_data**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline
2. Hop 1: Initial retrieval (k=7 docs) using the claim, followed by summarization
3. Hop 2: Generate refined query using claim + summary_1, retrieve k docs, summarize with context
4. Hop 3: Generate final query using claim + both summaries, retrieve k docs
5. Returns all 21 documents (7 per hop) as retrieved_docs

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from supporting_facts) are present in the retrieved documents (max 21). Returns True if gold titles are a subset of found titles, False otherwise. This evaluates the system's ability to successfully retrieve all relevant documents through multi-hop reasoning.

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

