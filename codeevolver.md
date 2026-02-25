PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-entity parallel document retrieval system with self-verification for fact-checking claims from the HoVer (Fact Extraction and VERification over Unstructured and Structured information) dataset. The system uses query decomposition to retrieve documents about all entities mentioned in multi-entity claims, performs gap detection to identify missing coverage, executes targeted retrieval to fill gaps, then applies two-stage reranking to select the most relevant documents.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline wrapper that initializes the ColBERTv2 retrieval model and orchestrates the program execution. Inherits from LangProBeDSPyMetaProgram and dspy.Module.
- **HoverMultiHop** (hover_program.py): Core retrieval logic implementing parallel multi-entity query decomposition with self-verification and targeted gap-filling. Decomposes claims into 2 focused sub-queries, retrieves k=25 documents per sub-query (up to 50 total), detects coverage gaps, performs one targeted retrieval (k=25) to fill gaps if needed, scores each document with chain-of-thought reasoning, deduplicates by title, and selects top 21 unique documents by relevance score. Total query limit: 3 (2 initial + 1 targeted).
- **ClaimDecomposition** (hover_program.py): DSPy signature that takes a claim and outputs 2-3 focused sub-queries targeting different entities or concepts within the claim for parallel retrieval.
- **RelevanceScorer** (hover_program.py): DSPy ChainOfThought signature that scores each document's relevance to the original claim on a 1-10 scale with reasoning, enabling intelligent reranking.
- **MissingEntityDetector** (hover_program.py): DSPy ChainOfThought signature that analyzes the claim and retrieved document titles to identify 1-2 specific entities or facts from the claim that are poorly covered by the current documents.
- **TargetedQueryGenerator** (hover_program.py): DSPy ChainOfThought signature that takes the claim and missing entities/facts and generates 1 highly specific query to find documents about those missing elements.
- **hoverBench** (hover_data.py): Dataset loader that filters HoVer dataset examples to only include 3-hop cases, formats them as DSPy examples with claims and supporting facts.
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are present in the retrieved documents (max 21 documents).

**Data Flow**:
1. Input claim is decomposed into 2 focused sub-queries targeting different entities/concepts
2. Each sub-query retrieves k=25 documents in parallel (up to 50 total documents)
3. Gap detection phase: Extract document titles and use MissingEntityDetector to identify 1-2 specific entities/facts from the claim that are poorly covered
4. Targeted retrieval phase (if gaps detected): Use TargetedQueryGenerator to create a specific query, then retrieve k=25 additional documents to fill gaps
5. All retrieved documents (initial + gap-filling) are combined into a single pool
6. Each document is scored for relevance to the original claim using chain-of-thought reasoning (RelevanceScorer) on a 1-10 scale
7. Documents are deduplicated by normalized title
8. Top 21 unique documents by relevance score are selected and returned as retrieved_docs

**Metric**: discrete_retrieval_eval evaluates whether all gold supporting fact documents are subset of retrieved documents, enforcing a maximum of 21 retrieved documents.

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

