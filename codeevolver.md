PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system performs iterative retrieval across three hops with gap analysis, adaptive query generation, and score-based reranking to find the most relevant supporting documents for veracity assessment of claims.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the HoverMultiHop program
- **HoverMultiHop** (hover_program.py): Core DSPy module implementing the 3-hop retrieval logic with gap analysis, adaptive query generation, document scoring, and intelligent reranking
- **GapAnalysis, AdaptiveQueryGenerator, DocumentScorer** (hover_program.py): DSPy Signature classes for identifying information gaps, generating targeted queries, and scoring document relevance
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. Input claim → Hop 1: Adaptive query generation → Retrieve k=20 docs → Summarize top 7
2. Gap Analysis: Identify missing information based on Summary 1
3. Hop 2: Generate gap-targeted query → Retrieve k=20 docs → Summarize with context
4. Gap Analysis: Identify remaining information gaps from Summaries 1&2
5. Hop 3: Generate final gap-targeted query → Retrieve k=20 docs
6. Score all 60 retrieved documents using DocumentScorer with ChainOfThought
7. Deduplicate by normalized document title, rank by relevance score
8. Output: Top 21 highest-scoring unique documents as retrieved_docs prediction

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

