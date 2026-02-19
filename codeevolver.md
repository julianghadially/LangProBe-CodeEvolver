PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: HoverMultiHopPipeline is a hypothesis-driven document retrieval system with self-verification that retrieves relevant supporting documents for a given claim. The system explicitly decomposes claims into verifiable hypotheses, performs comprehensive initial retrieval, and adaptively performs targeted retrieval for missing information. The system is evaluated on its ability to retrieve all gold-standard documents that support a claim.

**Key Modules**:
1. **HoverMultiHopPipeline** (`hover_pipeline.py`): Top-level wrapper that initializes a ColBERTv2 retrieval model and orchestrates the HoverMultiHop program execution with the retrieval context.

2. **HoverMultiHop** (`hover_program.py`): Core hypothesis-driven retrieval logic implementing a 2-stage adaptive strategy:
   - **Stage 1 - Hypothesis Decomposition & Initial Retrieval**: Uses ClaimDecomposer to break claims into 2-3 atomic sub-hypotheses, generates a comprehensive query covering all hypotheses, and retrieves k=15 documents.
   - **Stage 2 - Self-Verification & Targeted Retrieval**: Uses HypothesisVerifier to assess whether retrieved documents support all hypotheses, identifies missing entities, and performs up to 2 additional targeted retrievals (k=3 each) to fill specific gaps. Acts as a quality gate with explicit feedback on coverage.

3. **ClaimDecomposer** (`hover_program.py`): DSPy Signature that decomposes claims into 2-3 atomic sub-hypotheses that need independent verification. Ensures all entities and facts requiring verification are explicitly identified upfront.

4. **HypothesisVerifier** (`hover_program.py`): DSPy Signature that verifies which hypotheses are supported by retrieved documents, outputs verification status, confidence score (0-1), and a list of specific missing entities that need additional retrieval.

5. **Data Module** (`hover_data.py`): Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples with claims and supporting facts.

6. **Evaluation Metric** (`hover_utils.py`): The `discrete_retrieval_eval` function checks if all gold supporting document titles are present in the retrieved documents (maximum 21 documents).

**Data Flow**:
Claim → ClaimDecomposer (identify sub-hypotheses) → Generate comprehensive query → Initial retrieval (k=15) → HypothesisVerifier (assess coverage + identify gaps) → [If needed] Targeted retrieval 1 (k=3) → [If needed] Targeted retrieval 2 (k=3) → Return all documents (15 + up to 6 = max 21 total) → Evaluate against gold supporting facts using subset matching.

**Metric**: Binary success metric that returns True if all gold-standard supporting document titles are found within the top 21 retrieved documents.

**Key Innovation**: The self-verification loop provides explicit feedback on which hypotheses lack evidence, enabling targeted retrieval to fill specific gaps rather than hoping sequential hops randomly discover bridging documents. This addresses the root cause of missing documents in multi-hop reasoning tasks.

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

