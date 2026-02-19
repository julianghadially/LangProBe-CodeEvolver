PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a parallel multi-perspective multi-hop document retrieval system for the HOVER dataset that verifies factual claims by retrieving relevant supporting documents through iterative search hops with diverse query strategies.

**Key Modules**:
- **HoverMultiHopPipeline**: Top-level wrapper that initializes the ColBERTv2 retrieval model and delegates to the core program. Serves as the evaluation entry point.
- **HoverMultiHop**: Core multi-hop retrieval program that performs 3 retrieval hops. Hop 1 uses direct retrieval, while Hops 2 and 3 use parallel multi-perspective query generation (entity-based, contrastive, and relational) to explore diverse semantic directions. Includes novelty-based deduplication to maximize diversity.
- **Query Signature Classes**: Three specialized DSPy signatures for diverse query generation:
  - `EntityBasedQuery`: Focuses on extracting and querying named entities (people, places, organizations)
  - `ContrastiveQuery`: Focuses on what's missing, contradictory, or needs verification
  - `RelationalQuery`: Focuses on relationships and connections between concepts
- **hover_data.py**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=7 documents directly from claim, generate summary_1
3. Hop 2: Generate 3 diverse queries in parallel (entity, contrastive, relational) using claim + summary_1. Retrieve k=10 docs per query (30 total). Use novelty scoring to select top 7 most diverse docs that differ from hop 1. Generate summary_2.
4. Hop 3: Generate 3 diverse queries in parallel with updated context (claim + summary_1 + summary_2). Retrieve k=10 docs per query (30 total). Use novelty scoring to select top 7 most diverse docs from hops 1+2.
5. Return combined 21 documents (7 from each hop)
6. Evaluation compares retrieved document titles against ground truth supporting_facts

**Novelty-Based Selection**: Each hop's document selection uses a scoring mechanism that combines title uniqueness (70% weight) and content diversity (30% weight) to maximize exploration of different information sources while avoiding redundancy.

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

