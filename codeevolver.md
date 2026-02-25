PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements an Entity-Relationship Query Decomposition strategy for the HoVer (Hover-nlp) claim verification benchmark using DSPy. The system uses targeted entity-aware retrieval to find supporting documents for fact-checking claims that require evidence from multiple sources.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline that implements the Entity-Relationship Query Decomposition strategy. It extracts entities and relationships from claims, generates 3 targeted queries focused on different aspects (primary entity, secondary entity, connecting relationship), retrieves documents, deduplicates by title, and ranks by entity coverage. Initializes ColBERTv2 retrieval model and serves as the entry point for evaluation.

**HoverMultiHop** (`hover_program.py`): Legacy 3-hop iterative retrieval strategy (no longer used in forward() method). Each hop retrieves k=7 documents and uses Chain-of-Thought prompting to summarize findings.

**hover_utils**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting documents are found within the top 21 retrieved documents.

**hover_data**: Loads and preprocesses the HoVer dataset, filtering for 3-hop examples and formatting them for DSPy evaluation.

## Data Flow
1. Input claim enters via `HoverMultiHopPipeline.forward()`
2. **Entity Decomposition**: Extract all named entities (people, places, organizations, works, dates) and identify factual relationships using `DecomposeClaimEntities` signature
3. **Targeted Query Generation**: Generate 3 highly specific queries using `GenerateTargetedQueries`: (a) primary entity with key attributes, (b) secondary entity with relationships, (c) connecting relationship or comparison point
4. **Parallel Retrieval**: Retrieve k=11 documents per query (33 total documents)
5. **Deduplication**: Remove duplicate documents by title, keeping only unique entries
6. **Entity Coverage Ranking**: Score documents by counting entity/relationship mentions using `RankByEntityCoverage`, keep top 21
7. Return final ranked 21 documents as `retrieved_docs`

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

