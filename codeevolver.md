PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop document retrieval system for the HOVER dataset that verifies factual claims by retrieving relevant supporting documents through iterative search hops.

**Key Modules**:
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model and delegates to the core program. Serves as the evaluation entry point.
- **HoverMultiHop**: Core multi-hop retrieval program that performs 3 sequential retrieval hops using a diversified query strategy to ensure better coverage across entity types, locations, and connecting facts.
- **DiverseQueryGenerator**: DSPy signature that generates 3 diverse queries targeting (1) entities/people, (2) locations/places, and (3) events/works/connections. Takes the claim and previously retrieved document titles to avoid redundancy.
- **hover_data.py**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=7 documents directly from claim
3. Hop 2: Generate 2 diverse queries (entity and location) using claim + hop1 titles, retrieve k=4 per query (8 docs total)
4. Hop 3: Generate 2 diverse queries (entity and connection) using claim + hop1+hop2 titles, retrieve k=3 per query (6 docs total)
5. Return combined 21 documents (7+8+6 hops)
6. Evaluation compares retrieved document titles against ground truth supporting_facts

**Optimization Metric**: `discrete_retrieval_eval` returns True if all gold supporting document titles (normalized) are present in the top 21 retrieved documents, measuring retrieval recall success.

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

