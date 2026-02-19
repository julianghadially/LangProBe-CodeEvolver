PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a confidence-based iterative multi-hop document retrieval system for the HOVER dataset that verifies factual claims by retrieving relevant supporting documents through adaptive search hops with self-verification feedback loops.

**Key Modules**:
- **HoverMultiHopPipeline**: Top-level pipeline that implements iterative confidence-based retrieval architecture. Contains three new DSPy modules: ConfidenceScorer (scores passage relevance 0-1), AdaptiveQueryRefiner (generates improved queries when confidence is low), and ListwiseReranker (selects top 21 from all retrieved passages). Serves as the evaluation entry point.
- **HoverMultiHop**: Legacy core multi-hop retrieval program (no longer used in evaluation).
- **hover_data.py**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=30 documents from claim, score with ConfidenceScorer. If avg confidence < 0.6 and searches < 3, refine query with AdaptiveQueryRefiner and retrieve k=30 more (still hop 1). Generate summary.
3. Hop 2: Generate query from claim + summary_1, retrieve k=30 docs, score passages. If confidence low and searches remain, refine and retrieve more. Generate summary_2.
4. Hop 3: Generate query from claim + summaries, retrieve k=30 docs, score passages. If confidence low and searches remain, refine and retrieve more.
5. Accumulate all passages from all hops (up to 180 docs max if all hops refined) with confidence scores
6. ListwiseReranker selects top 21 most relevant documents from full pool based on claim relevance
7. Return final 21 documents
8. Evaluation compares retrieved document titles against ground truth supporting_facts

**Optimization Metric**: `discrete_retrieval_eval` returns True if all gold supporting document titles (normalized) are present in the top 21 retrieved documents, measuring retrieval recall success.

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

