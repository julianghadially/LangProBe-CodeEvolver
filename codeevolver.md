PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Overview
This program implements Backward Chaining with Constraint Propagation for multi-hop document retrieval on the HoVer claim verification benchmark using DSPy. The system fundamentally changes retrieval reasoning from forward decomposition to backward constraint satisfaction. It generates competing hypotheses about what evidence would prove/disprove claims, extracts specific constraints (entities, dates, relationships, negations), performs targeted backward searches (max 3 searches, k=7 per search), and scores documents by constraint satisfaction with diversity penalties. Returns top 21 documents.

## Key Modules

**HoverMultiHopPipeline** (`hover_pipeline.py`): Main pipeline implementing backward chaining with constraint propagation. Generates 3-4 hypotheses, extracts constraints from claim, performs backward searches on top 2-3 hypotheses (max 3 total searches with k=7 retrievals each), deduplicates by title, scores by constraint satisfaction + diversity penalty, returns top 21. Entry point for evaluation.

**HypothesisGenerator** (`hover_pipeline.py`): Signature generating 3-4 competing hypotheses about what supporting documents would prove/disprove the claim.

**ConstraintExtractor** (`hover_pipeline.py`): Signature extracting specific constraints (entity names, dates, relationships, negations) from the claim that documents must satisfy.

**BackwardQuery** (`hover_pipeline.py`): Signature generating targeted queries designed to find documents satisfying specific constraints and testing hypotheses.

**ConstraintSatisfactionScorer** (`hover_pipeline.py`): ChainOfThought module scoring documents (0-10) by counting how many extracted constraints they satisfy.

**hover_utils**: Contains `discrete_retrieval_eval` metric for recall@21 evaluation.

**hover_data**: Loads HoVer dataset with 3-hop examples.

## Data Flow
1. Input claim enters `HoverMultiHopPipeline.forward()`
2. **Step 1**: Generate 3-4 competing hypotheses about what evidence would prove/disprove claim → Extract specific constraints (entities, dates, relationships, negations) from claim
3. **Step 2**: Select top 2-3 hypotheses → For each hypothesis, generate targeted backward queries satisfying constraints → Perform max 3 total searches (distributed across hypotheses) with k=7 retrievals each → Deduplicate by title as retrieved
4. **Step 3**: Score each document by constraint satisfaction (0-10) → Apply diversity penalty (0-3) to penalize documents too similar to already-selected ones → Final score = constraint_score - diversity_penalty
5. **Step 4**: Sort by final score descending → Return top 21 documents

## Metric
The `discrete_retrieval_eval` metric computes recall@21: whether all gold supporting document titles are in the retrieved set. The backward chaining with constraint propagation architecture maximizes recall by generating competing hypotheses about evidence, extracting explicit constraints from claims, performing targeted backward searches to find documents satisfying those constraints, and using constraint satisfaction scoring with diversity penalties to select the most relevant 21 documents.

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

