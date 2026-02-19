PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a multi-hop document retrieval system that retrieves relevant supporting documents for a given claim using iterative retrieval and summarization steps. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level wrapper that initializes a ColBERTv2 retrieval model and orchestrates the HoverMultiHop program execution with the retrieval context.

2. **HoverMultiHop** (`hover_program.py`): Core adaptive multi-hop retrieval logic implementing confidence-weighted retrieval with coverage tracking. Uses up to 3 hops with dynamic k values (10/8/5) based on coverage confidence scores. Includes CoverageAssessment and AdaptiveQueryGenerator signatures for self-correcting feedback loops that identify missing entities and generate targeted queries to maximize coverage diversity while avoiding redundant documents.

3. **CoverageAssessment** (signature in `hover_program.py`): DSPy signature that evaluates how well retrieved passages cover the claim, outputting confidence scores (0-1), missing entities, and coverage summaries to guide subsequent retrievals.

4. **AdaptiveQueryGenerator** (signature in `hover_program.py`): DSPy signature that generates focused search queries targeting missing entities and uncovered aspects based on coverage feedback and previous query history.

5. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

6. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → Hop1 (k=10 broad retrieval) → Coverage assessment (confidence + missing entities) → Adaptive k selection (10/8/5 based on confidence) → Hop2 (targeted query for missing entities) → Coverage reassessment → Early stopping if confidence ≥0.9 → Hop3 (final targeted retrieval if needed) → Deduplication via seen_titles set → Ensure exactly 21 unique documents → Evaluate against gold supporting facts using subset matching.

**Adaptive Retrieval Mechanism**: After each hop, the system assesses coverage confidence. If confidence < 0.3, uses k=10; if < 0.6, uses k=8; else k=5. Queries are generated to explicitly target missing_entities, creating a feedback loop that corrects for gaps in coverage. Documents are deduplicated by title, and early stopping occurs if confidence ≥ 0.9 after 2 hops.

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

