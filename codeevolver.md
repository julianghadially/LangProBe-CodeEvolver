PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system performs three iterative retrieval hops to find relevant supporting documents for a given claim, using a ColBERTv2 retriever and an adaptive query portfolio architecture with LLM-based query selection to progressively refine queries.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level pipeline that implements adaptive query portfolio architecture with LLM-driven query selection. At each hop, it generates 3-5 diverse candidate queries using different strategies (entity-focused, relation-focused, temporal-focused, etc.), scores them, and selects the top-1 query to execute for retrieval.
- **QueryPortfolioGenerator**: DSPy signature that generates 3-5 diverse candidate queries using different retrieval strategies
- **QueryConfidenceScorer**: DSPy signature that scores each candidate query's expected retrieval utility given the claim and context
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Hop 1: Generate 3-5 diverse candidate queries using QueryPortfolioGenerator (no prior context), score them with QueryConfidenceScorer, select top-1 query, retrieve k=7 documents, summarize results
3. Hop 2: Generate 3-5 candidate queries using claim + summary_1 as context, score and select top-1 query, retrieve k=7 documents, create summary_2
4. Hop 3: Generate 3-5 candidate queries using claim + both summaries as context, score and select top-1 query, retrieve final k=7 documents
5. Return all 21 documents (3 hops × 7 docs each) as retrieved_docs

The adaptive query portfolio architecture maintains the 3-search constraint (3 retrievals total, 7 docs each) while exploring multiple retrieval strategies per hop through LLM-based query generation and scoring. This allows the system to consider diverse query approaches before committing to a single retrieval operation at each hop.

**Metric**: The discrete_retrieval_eval metric checks if all gold-standard supporting document titles (from supporting_facts) are present in the top 21 retrieved documents. Success requires 100% recall of gold documents within the 21-document limit. Documents are compared using normalized text matching on title keys.

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

