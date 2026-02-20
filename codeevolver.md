PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This system performs entity-focused document retrieval for claim verification using the HoVer dataset. It retrieves relevant supporting documents by first extracting key entities from the claim, then generating focused queries for each entity and using LLM-based reranking to select the most relevant documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing entity-focused retrieval strategy. Initializes ColBERTv2 retrieval model and coordinates the entity extraction, focused query generation, retrieval, and reranking process.
- `EntityExtractor`: DSPy ChainOfThought module that extracts 2-4 key entities/topics from the claim (e.g., person names, organizations, titles, events)
- `FocusedQueryGenerator`: DSPy ChainOfThought module that generates highly specific search queries for each extracted entity
- `TopKReranker`: LLM-based scoring system that evaluates document relevance on a 0-10 scale and selects top 7 documents per query
- `hover_utils.py`: Contains the evaluation metric `discrete_retrieval_eval` that validates retrieval quality
- `hover_data.py`: Benchmark class managing the HoVer dataset, filtering for 3-hop examples

**Data Flow**:
1. Extract 2-4 key entities/topics from the input claim using EntityExtractor
2. Limit to maximum 3 entities to respect query constraints
3. For each entity:
   a. Generate a focused search query using FocusedQueryGenerator
   b. Retrieve k=100 documents using dspy.Retrieve
   c. Score each document's relevance to the original claim (0-10 scale) using LLM
   d. Select top 7 documents based on relevance scores
4. Combine all retrieved documents from all entities (up to 21 documents)
5. Deduplicate documents to remove any overlaps
6. Return final set of unique documents (up to 21 total)

**Metric**: `discrete_retrieval_eval` checks if all gold-standard supporting document titles are present in the retrieved set (up to 21 documents). Returns True only when all required documents are successfully retrieved, evaluating as a strict subset match.

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

