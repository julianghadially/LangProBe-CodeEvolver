{
  "architecture": "PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPredictPipeline\nMETRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval\n\n## Purpose\nThis is a multi-hop fact-checking system for the HoVer (Fact Extraction and VERification over Unstructured and Structured information) benchmark. The system retrieves supporting documents across three iterative reasoning hops to verify factual claims.\n\n## Key Modules\n\n**HoverMultiHopPredictPipeline** (hover_pipeline.py): Top-level orchestrator that initializes a ColBERTv2 retrieval model and wraps the core prediction program. Sets up the retrieval context with a remote ColBERT service.\n\n**HoverMultiHopPredict** (hover_program.py): Core multi-hop reasoning engine implementing a three-hop retrieval strategy. Each hop retrieves k=7 documents, creates summaries, and generates queries for subsequent hops using DSPy predictors.\n\n**hoverBench** (hover_data.py): Dataset loader that filters the HoVer dataset to examples requiring exactly 3 supporting documents (3-hop reasoning). Prepares train/validation splits with claim-label-supporting_facts structure.\n\n**discrete_retrieval_eval** (hover_utils.py): Evaluation metric checking if all gold supporting document titles are found within the top 21 retrieved documents (across all hops).\n\n## Data Flow\n1. Input claim → Hop 1: retrieve k documents, summarize\n2. Summary 1 + claim → generate query → Hop 2: retrieve k documents, summarize\n3. Summaries 1&2 + claim → generate query → Hop 3: retrieve k documents\n4. Return concatenated docs from all hops (21 total)\n5. Metric evaluates if gold document titles ⊆ retrieved document titles\n\n## Optimization Target\nThe metric optimizes for recall of supporting documents: success requires retrieving ALL gold-labeled supporting documents within the 21-document budget across three reasoning hops."
}

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

