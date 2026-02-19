PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop document retrieval system that retrieves relevant supporting documents for a given claim using entity-focused query decomposition and relevance-based reranking. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level wrapper that initializes a ColBERTv2 retrieval model and orchestrates the HoverMultiHop program execution with the retrieval context.

2. **HoverMultiHop** (`hover_program.py`): Core multi-hop retrieval logic implementing entity-focused query decomposition with relevance-based reranking. Uses exactly 3 search operations across 3 hops: Hop1 retrieves k=10 docs for the original claim; Hop2 extracts 2-4 key entities and performs 1-2 entity-specific searches (k=8-10 each); Hop3 performs gap-filling retrieval (k=15) if needed. Retrieves up to 50 total documents, then reranks by keyword/entity relevance to return top 21.

3. **EntityExtraction** (signature in `hover_program.py`): DSPy signature that extracts 2-4 key named entities or topics from the claim, focusing on people, organizations, locations, events, or specific concepts critical for verification.

4. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

5. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Extract keywords → Hop1 (k=10 broad retrieval on claim) → Entity extraction (2-4 entities) → Hop2 (1-2 entity-focused queries, k=8-10 each, max 3 searches total) → Hop3 (optional gap-filling query k=15 if <3 searches used and <50 docs) → Collect up to 50 unique documents → Relevance-based reranking (score by keyword/entity presence in title/content) → Return top 21 highest-scoring documents → Evaluate against gold supporting facts using subset matching.

**Entity-Focused Retrieval Mechanism**: The system extracts 2-4 key entities from the claim using the EntityExtraction signature. In Hop2, it generates 1-2 entity-specific queries by combining each entity with the original claim context, retrieving k=8-10 documents per entity query. This ensures critical entity-related documents are retrieved early. Hop3 performs gap-filling using remaining entities or combined entity queries. All documents are deduplicated by title throughout.

**Relevance-Based Reranking**: After collecting up to 50 documents across all hops, each document is scored based on keyword and entity presence. Title matches receive higher weights (5.0 for entities, 3.0 for keywords) than content matches (2.0 for entities, 1.0 for keywords). Documents are sorted by score in descending order, and the top 21 are returned.

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

