PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system uses a Cross-Attention Multi-Path Retrieval architecture to find relevant supporting documents for a given claim, leveraging ColBERTv2 retrieval with aspect-based query specialization and attention-based fusion to avoid lossy sequential summarization.

**Key Modules**:
- **HoverMultiHopPipeline**: The main pipeline implementing Cross-Attention Multi-Path Retrieval with aspect extraction, parallel specialized searches, and cross-attention reranking
- **ClaimAspectExtractor**: DSPy signature that analyzes claims and extracts 3 distinct retrieval perspectives (primary entities, relationships/connections, contextual/temporal info)
- **AspectQueryGenerator**: DSPy signature that generates specialized search queries for each aspect
- **CrossAttentionReranker**: DSPy signature that uses chain-of-thought to select 21 documents from 75 candidates, considering relevance, diversity across aspects, and cross-document coherence
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset, filtering for 3-hop examples
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Aspect Extraction: Use ClaimAspectExtractor to identify 3 retrieval perspectives (entities, relationships, context)
3. Parallel Retrieval: For each aspect, generate specialized query via AspectQueryGenerator and retrieve k=25 documents (total 75 docs)
4. Cross-Attention Reranking: CrossAttentionReranker analyzes all 75 documents grouped by aspect and selects exactly 21 documents that provide complementary evidence
5. Return selected 21 documents as retrieved_docs, ensuring diversity and coherence

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

