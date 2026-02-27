PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

This program implements a multi-hop document retrieval system for the HoVer (Hover-nlp) fact verification task using DSPy. The system retrieves supporting documents for a given claim through iterative retrieval and summarization.

**Key Modules:**

1. **HoverMultiHopPipeline** (`hover_pipeline.py`): The top-level pipeline wrapper that initializes the ColBERTv2 retriever with a remote API endpoint and orchestrates the overall execution flow.

2. **HoverMultiHop** (`hover_program.py`): The core multi-hop retrieval program that performs three sequential retrieval hops with gap analysis:
   - Gap Analysis: Analyzes the claim upfront to identify key entities, information needs, and search strategy
   - Hop 1: Generates a targeted query using gap analysis outputs (key_entities, information_needed, search_strategy), retrieves k=7 documents
   - Hop 2: Generates a refined query using claim + gap analysis + summary from hop 1, retrieves k=7 more documents
   - Hop 3: Generates another query using claim + gap analysis + both previous summaries, retrieves k=7 additional documents
   - Returns all 21 documents (3 hops × 7 documents each)

3. **hover_data.py**: Manages dataset loading from the HoVer benchmark, filtering for 3-hop examples and creating DSPy Example objects.

4. **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` which checks if all gold supporting document titles are present in the retrieved documents (subset match).

**Data Flow:**
Claim → Gap Analysis (key_entities, information_needed, search_strategy) → Hop1 Query Generation (with analysis) → Hop1 Retrieve → Summarize → Hop2 Query Generation (with analysis + summary_1) → Hop2 Retrieve → Summarize → Hop3 Query Generation (with analysis + summary_1 + summary_2) → Hop3 Retrieve → Combined 21 Documents

**Metric:** The `discrete_retrieval_eval` metric returns True if all gold standard supporting document titles are found within the predicted retrieved documents (max 21), using normalized text matching.

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

