PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This is a multi-hop document retrieval system for fact-checking claims from the HoVer dataset. The system uses Query Expansion with Semantic Clustering to find relevant supporting documents for a given claim, combining diverse retrieval strategies with intelligent post-retrieval filtering.

**Key Modules**:
- **HoverMultiHopPipeline**: The top-level module implementing query expansion, LLM-based document scoring, and semantic clustering
- **EntityExtractor**: DSPy signature for extracting named entities and generating entity-focused queries
- **RelationshipQueryGenerator**: DSPy signature for generating relationship-focused queries
- **FactVerificationQueryGenerator**: DSPy signature for generating fact-verification queries
- **DocumentScorer**: DSPy signature for scoring document relevance with reasoning (0-10 scale)
- **HoverMultiHop**: The original 3-hop retrieval program (kept for reference but not used)
- **hover_data.py**: Data loading and preprocessing from the HoVer dataset
- **hover_utils.py**: Contains the evaluation metric and document counting utilities

**Data Flow**:
1. Input claim enters HoverMultiHopPipeline.forward()
2. Generate three complementary queries:
   - Entity-focused: Extract named entities and create query about them
   - Relationship-focused: Target connections between entities
   - Fact-verification: Focus on specific claims to verify
3. Retrieve k=30 documents per query (90 total), then deduplicate
4. Score all unique documents using LLM-based DocumentScorer (0-10 scale with reasoning)
5. Select top 35 highest-scored documents
6. Cluster top 35 into 3 semantic clusters using sentence embeddings (all-MiniLM-L6-v2)
7. Select final 21 documents ensuring at least 2 from each cluster for diversity
8. Return 21 documents as retrieved_docs

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

