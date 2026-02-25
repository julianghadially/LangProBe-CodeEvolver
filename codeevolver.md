PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements a parallel multi-perspective document retrieval system for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system generates diverse queries from different analytical perspectives and executes parallel retrievals to find supporting documents for fact-checking claims that require evidence from multiple sources.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline that implements a parallel multi-perspective query generation strategy. Initializes the ColBERTv2 retrieval model, generates 3 diverse queries simultaneously (entity-focused, relationship-focused, and contextual), executes parallel retrievals with k=11 each, and applies frequency-based deduplication to rank documents that appear in multiple query results higher. Serves as the entry point for the evaluation framework.

**GenerateDiverseQueries** (DSPy Signature in `hover_pipeline.py`): Signature that generates exactly 3 diverse, non-overlapping search queries from a claim: (1) entity_query focused on named entities (people, places, organizations, dates), (2) relationship_query focused on connections/comparisons between entities, and (3) contextual_query focused on broader domain/category context.

**HoverMultiHop** (`hover_program.py`): Legacy retrieval logic (not currently used) that implemented a 3-hop iterative retrieval strategy with Chain-of-Thought summarization.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. Generate 3 diverse queries in parallel using `GenerateDiverseQueries` signature (entity, relationship, contextual perspectives)
3. Execute 3 parallel retrieval calls with k=11 each (~33 total documents)
4. Combine all retrieved documents and deduplicate using frequency-based scoring (documents appearing in multiple query results rank higher)
5. Truncate to exactly 21 documents and return as `retrieved_docs`

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles from `supporting_facts` are present in the retrieved set. Success requires the retrieval pipeline to discover all necessary evidence documents within the 21-document budget.

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

