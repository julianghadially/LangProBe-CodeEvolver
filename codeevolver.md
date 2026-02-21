PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a gap-analysis-based multi-hop document retrieval system for fact-checking claims using the HoVer (Hover-nlp) dataset. The system extracts entities from claims, performs initial retrieval, analyzes coverage gaps, targets missing entities with follow-up queries, and uses entity-aware reranking to find the most relevant supporting documents for veracity assessment.

**Key Modules**:
- **HoverMultiHopPipeline** (hover_pipeline.py): Top-level pipeline implementing gap analysis retrieval strategy with claim entity extraction, gap detection, targeted follow-up queries, and entity-aware reranking
- **ClaimEntityExtraction** (hover_pipeline.py): DSPy Signature for extracting 3-7 key named entities (people, places, organizations) directly from the claim text
- **GapAnalysis** (hover_pipeline.py): DSPy Signature for analyzing which claim entities are NOT well-covered in retrieved documents
- **TargetedQueryGenerator** (hover_pipeline.py): DSPy Signature for generating targeted search queries for missing/poorly-covered entities
- **EntityAwareReranker** (hover_pipeline.py): DSPy Signature for scoring and ranking documents with priority given to documents mentioning multiple claim entities
- **EntityExtraction** (hover_pipeline.py): Legacy DSPy Signature for extracting entities from documents (kept for compatibility)
- **EntityQueryGenerator** (hover_pipeline.py): Legacy DSPy Signature for entity-based query generation (kept for compatibility)
- **ListwiseReranker** (hover_pipeline.py): Legacy DSPy Signature for standard document reranking (kept for compatibility)
- **HoverMultiHop** (hover_program.py): Legacy DSPy module implementing 3-hop retrieval logic (not currently used)
- **hoverBench** (hover_data.py): Dataset handler that loads and filters HoVer dataset to 3-hop examples, creating train/test splits
- **discrete_retrieval_eval** (hover_utils.py): Evaluation metric that checks if all gold supporting document titles are retrieved (maximum 21 documents)

**Data Flow**:
1. **Stage 1 - Claim Entity Extraction**: Extract 3-7 key named entities (people, places, organizations) from the claim using LLM-based ClaimEntityExtraction module
2. **Stage 2 - Initial Retrieval**: Retrieve k=75 documents (range 50-100) using the original claim as query (Retrieval call #1)
3. **Stage 3 - Gap Analysis**: Analyze initial retrieved documents to identify which claim entities are poorly covered or missing using GapAnalysis module
4. **Stage 4 - Targeted Query Generation**: For each missing entity (up to 2), generate a targeted search query using TargetedQueryGenerator
5. **Stage 5 - Gap-Filling Retrieval**: For each targeted query, retrieve k=40 additional documents (range 30-50) to fill coverage gaps (Retrieval calls #2-3, max 3 total)
6. **Stage 6 - Document Merging**: Combine all retrieved documents (initial + gap-filling) and deduplicate by normalized document title
7. **Stage 7 - Entity-Aware Reranking**: Apply EntityAwareReranker using ChainOfThought reasoning to score documents, prioritizing those mentioning multiple claim entities, selecting top 21 most relevant documents
8. Output: Final 21 highest-ranked documents as retrieved_docs prediction

**Retrieval Constraints**: Maximum 3 retrieval calls total (1 initial + up to 2 gap-filling)

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

