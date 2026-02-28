PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is an entity-aware document retrieval system for fact-checking claims using the HoVer (Hop Verification) dataset. The system performs systematic entity-based retrieval to ensure all claim components are addressed, using a four-stage pipeline that extracts entities, retrieves documents per entity, analyzes coverage gaps, and selects the most relevant documents.

**Key Modules**:
- `HoverMultiHopPipeline`: Top-level module implementing entity-aware retrieval pipeline with four DSPy signatures (EntityExtraction, EntityQueryGenerator, GapAnalysis, FillInQueryGenerator)
- `HoverMultiHop`: Legacy 3-hop retrieval program (no longer used)
- `hover_data.py`: Data loader that filters HoVer dataset to 3-hop examples (claim-fact pairs requiring 3 documents)
- `hover_utils.py`: Evaluation utilities including the `discrete_retrieval_eval` metric

**Data Flow**:
1. EntityExtraction: Extracts key entities from the claim (people, places, organizations, events, concepts)
2. Entity Query Generation: Creates one focused query per entity (max 3 entities) and retrieves k=15 documents per query
3. Gap Analysis: Identifies which entities lack supporting documents in the retrieved set
4. Fill-in Retrieval: If gaps exist and queries remain available, generates one fill-in query targeting missing entities and retrieves k=15 more documents
5. Deduplication & Scoring: Deduplicates documents by title, scores each by counting entity mentions, and selects top 21 documents
6. Final output returns exactly 21 documents as `retrieved_docs`

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

