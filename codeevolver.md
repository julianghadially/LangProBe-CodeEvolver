PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is an advanced multi-hop document retrieval system using gap-aware iterative retrieval with utility-based reranking to retrieve relevant supporting documents for a given claim. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level pipeline implementing a sophisticated 3-hop retrieval strategy with gap analysis, deduplication, and utility-based reranking. Each hop retrieves k=40 candidate documents, deduplicates against previously seen titles, and reranks by utility (relevance, gap coverage, novelty) to select the top documents. After all hops, performs final reranking across all candidates to select the top 21 documents.

2. **GapAnalyzer** (within `hover_pipeline.py`): DSPy ChainOfThought module that identifies missing information from the claim not yet covered by retrieved documents, guiding subsequent hop queries.

3. **UtilityReranker** (within `hover_pipeline.py`): DSPy ChainOfThought module that scores documents based on: (a) relevance to claim, (b) coverage of identified information gaps, and (c) novelty compared to already-selected documents.

4. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

5. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Hop1 (retrieve 40, dedupe, rerank to ~10 best) → Summarize → GapAnalysis → Generate Hop2 query targeting gaps → Hop2 (retrieve 40, dedupe globally, rerank to ~10) → Summarize → GapAnalysis → Generate Hop3 query targeting remaining gaps → Hop3 (retrieve 40, dedupe globally, rerank to ~10) → Final utility-based reranking across all unique candidates → Top 21 documents → Evaluate against gold supporting facts using subset matching.

**Key Features**:
- **Gap-Aware Retrieval**: After each hop, analyzes what information is still missing to guide subsequent queries
- **Adaptive Retrieval**: Retrieves 40 candidates per hop instead of fixed k=7, allowing better coverage
- **Utility-Based Reranking**: Scores documents on relevance, gap coverage, and novelty rather than just retrieval order
- **Global Deduplication**: Tracks document titles across all hops to avoid wasting slots on duplicates
- **Two-Stage Selection**: Per-hop reranking (~10 docs) followed by final cross-hop reranking (top 21)

**Metric**: Binary success metric that returns True if all gold-standard supporting document titles are found within the top 21 retrieved documents.

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

