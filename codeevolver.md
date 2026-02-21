PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a Diversity-Aware Iterative Retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system performs parallel diverse retrieval with coverage-aware gap filling to maximize entity and aspect coverage while maintaining the 21-document limit.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing Diversity-Aware Iterative Retrieval with four new DSPy Signatures: ClaimDecomposer, CoverageAnalyzer, GapQuery, and DiversityReranker. Initializes ColBERTv2 retrieval model and orchestrates the diversity-aware retrieval flow.
- **HoverMultiHop** (hover_program.py): Legacy 3-hop retrieval module (not currently used by pipeline)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Signature Classes**:
1. **ClaimDecomposer**: Decomposes claim into 2-3 sub-queries targeting different entities/aspects
2. **CoverageAnalyzer**: Identifies entities/concepts missing from current retrieved documents
3. **GapQuery**: Generates targeted queries to retrieve missing information
4. **DiversityReranker**: Selects top 21 most diverse and relevant documents using MMR-style scoring

**Data Flow**:
1. Input claim → Decompose into 2-3 diverse sub-queries (planning phase, not retrieval)
2. Hop 1: Retrieve k=10 documents per sub-query in parallel, deduplicate
3. Coverage Analysis: Identify missing entities/concepts from retrieved documents
4. Hop 2: Generate gap-filling query and retrieve k=15 documents
5. Combine and deduplicate all documents from both hops
6. Diversity Reranking: Apply MMR-style scoring to select final 21 most diverse and relevant documents
7. Output: 21 documents as retrieved_docs prediction

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

