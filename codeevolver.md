PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a Recursive Query Expansion system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system leverages retrieved document titles to guide subsequent entity-focused queries, ensuring multi-hop entity coverage while staying within the 3-search constraint and 21-document limit.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing Recursive Query Expansion with four DSPy Signatures: InitialQueryGenerator, EntityExtractor, EntityQueryGenerator, and RelevanceReranker. Initializes ColBERTv2 retrieval model and orchestrates entity-driven recursive retrieval flow.
- **HoverMultiHop** (hover_program.py): Legacy 3-hop retrieval module (not currently used by pipeline)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Signature Classes**:
1. **InitialQueryGenerator**: Generates initial search query from claim
2. **EntityExtractor**: Extracts 2-5 unexplored entity mentions from retrieved document titles
3. **EntityQueryGenerator**: Generates focused queries for specific entities
4. **RelevanceReranker**: Scores all documents by relevance to claim and selects top 21

**Data Flow**:
1. Input claim → Generate initial query
2. Hop 1: Retrieve k=15 documents with initial query
3. Entity Extraction: Extract entity mentions from retrieved document titles
4. Hop 2: Generate targeted queries for top 2-3 unexplored entities, retrieve k=10 docs each
5. Conditional Hop 3: If under 50 total docs, extract entities again and retrieve k=15 more docs
6. Deduplicate all documents across all hops
7. Relevance Reranking: Score all documents and select top 21 by relevance
8. Output: Top 21 documents as retrieved_docs prediction

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

