PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop document retrieval system that retrieves relevant supporting documents for a given claim using entity-focused query decomposition and iterative retrieval. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level module implementing entity-focused query decomposition strategy. First extracts 3-5 key named entities from the claim using ClaimEntityExtractor signature, then performs 3-hop retrieval. Hop 1 generates one focused query per entity (up to 3), retrieves k=50 docs per query (150 total), applies utility-based reranking (prioritizing docs found by multiple entity queries) to select top 7. Hops 2 and 3 follow the original strategy with k=7 each. Total: 21 documents.

2. **ClaimEntityExtractor** (`hover_pipeline.py`): DSPy Signature class that extracts 3-5 key named entities (people, places, organizations, titles) from the claim for focused retrieval.

3. **HoverMultiHop** (`hover_program.py`): Core multi-hop retrieval logic implementing a 3-hop strategy. Each hop retrieves k=7 documents, uses ChainOfThought to generate summaries and subsequent queries, building upon previous hop results. (Note: Currently bypassed in favor of direct implementation in HoverMultiHopPipeline)

4. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

5. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Entity Extraction (3-5 entities) → Hop1: Entity-focused retrieval (3 queries × k=50, rerank to top 7) → Summarize → Generate Hop2 query → Hop2 retrieval (k=7) → Summarize → Generate Hop3 query → Hop3 retrieval (k=7) → Concatenate all documents (21 total) → Evaluate against gold supporting facts using subset matching.

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

