PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system uses a hybrid retrieval strategy that combines broad contextual queries with targeted entity-specific queries to find supporting documents relevant to a given claim.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level pipeline implementing hybrid retrieval strategy with entity extraction, contextual query generation, and duplicate-aware reranking
- `ExtractKeyEntities`: DSPy signature for extracting proper nouns, names, dates, and specific terms from claims
- `GenerateContextualQuery`: DSPy signature for generating broad contextual queries covering the main claim relationship
- `HoverMultiHop`: Legacy core program implementing 3-hop retrieval logic (not currently used by HoverMultiHopPipeline)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. Input claim is processed by entity extractor to identify key entities (proper nouns, names, dates, specific terms)
2. A broad contextual query is generated covering the main claim relationship
3. Documents (k=10) are retrieved using the contextual query
4. Top 2-3 most specific entities are selected (prioritizing uncommon names, company names, battle names, work titles)
5. For each top entity, k=10 documents are retrieved using the entity name as a direct query (max 3 total queries)
6. All retrieved documents are deduplicated by title and reranked based on:
   - Exact entity name matches (highest priority, 100 points per match)
   - Claim keyword coverage (10 points per keyword)
   - Retrieval rank position (0-30 point penalty)
7. Top 21 unique documents after reranking are returned as `retrieved_docs`

**Metric**: `discrete_retrieval_eval` checks if all gold supporting document titles (from `supporting_facts`) are present in the retrieved documents (max 21). Returns True if gold titles are a subset of retrieved titles after normalization.

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

