PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements a multi-hop document retrieval system for the HOVER dataset that verifies factual claims by retrieving relevant supporting documents through iterative search hops.

**Key Modules**:
- **HoverMultiHopPipeline**: Top-level wrapper implementing a two-stage sentence-level retrieval architecture. Initializes ColBERTv2 retrieval model, delegates query generation to HoverMultiHop, and performs sentence extraction and re-ranking. Serves as the evaluation entry point.
- **SentenceRelevanceScorer**: DSPy module that uses LLM to extract and score the 2-3 most relevant sentences from each retrieved document, outputting a relevance score (0.0-1.0) and the best sentences.
- **HoverMultiHop**: Core multi-hop retrieval program that performs 3 sequential retrieval hops, using Chain-of-Thought reasoning to generate queries and summarize retrieved passages at each step.
- **hover_data.py**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline.forward()
2. **Stage 1 - Broad Retrieval**: Perform 3-hop retrieval with k=50 documents per query:
   - Hop 1: Retrieve k=50 documents directly from claim, generate summary from top 7
   - Hop 2: Generate new query from claim + summary_1, retrieve k=50 more docs, generate summary_2 from top 7
   - Hop 3: Generate final query from claim + summary_1 + summary_2, retrieve k=50 final docs
   - Combine all retrieved documents (up to 150 total)
3. **Stage 2 - Sentence Extraction & Re-ranking**: For each of the 150 documents:
   - Extract document title and text
   - Use SentenceRelevanceScorer to identify 2-3 most relevant sentences and assign relevance score
   - Format as "title | extracted_sentences"
4. Sort all documents by relevance score (descending) and select top 21
5. Return top 21 document-sentence pairs as final retrieved_docs
6. Evaluation compares retrieved document titles against ground truth supporting_facts

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

