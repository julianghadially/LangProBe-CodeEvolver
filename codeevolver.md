PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system performs entity-based multi-query retrieval to find supporting documents for veracity assessment of claims.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing Named Entity Extraction + Entity-Based Multi-Query Retrieval architecture with three DSPy signature classes: EntityExtractor, EntityQueryGenerator, and EntityCoverageReranker
- **HoverMultiHop** (hover_program.py): Legacy DSPy module implementing the 3-hop retrieval logic (still available but not used in current pipeline)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. Input claim → EntityExtractor: Extract up to 3 key named entities (people, organizations, locations, events)
2. For each entity → EntityQueryGenerator: Create focused search query for the entity in context of claim
3. Execute queries → Retrieve k=70 documents per query (max 3 queries = up to 210 docs)
4. Deduplicate by document title → Combine unique documents from all queries
5. EntityCoverageReranker → Score each document based on entity coverage + semantic relevance to claim
6. Output: Top 21 highest-scored documents as retrieved_docs prediction

**Metric**: discrete_retrieval_eval compares normalized gold document titles against retrieved document titles, returning True if all gold titles are found within the retrieved set (subset check).

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

