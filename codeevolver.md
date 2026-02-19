PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a question decomposition-based document retrieval system for fact-checking claims from the HoVer dataset. The system decomposes claims into atomic sub-questions, performs targeted retrieval for each sub-question, and uses coverage-based scoring to select the most relevant documents that answer all sub-questions.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level module implementing the question decomposition and coverage-based retrieval architecture
- **QuestionDecomposer**: DSPy signature that decomposes a claim into up to 3 atomic sub-questions
- **DocumentScorer**: DSPy signature that scores documents based on their relevance to answering all sub-questions
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. QuestionDecomposer (using ChainOfThought) decomposes claim into up to 3 atomic sub-questions
3. For each sub-question, retrieve k=30 documents using ColBERTv2 (max 3 searches, respecting 3-search constraint)
4. Deduplicate all retrieved documents to create a unique document pool
5. DocumentScorer (using ChainOfThought) evaluates each unique document for coverage of all sub-questions, assigning a relevance score (0.0-1.0)
6. Sort documents by aggregate relevance scores in descending order
7. Return top 21 highest-scoring documents as retrieved_docs

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

