PARENT_MODULE_PATH: langProBe.hover.hover_pipeline.HoverMultiHopPipeline
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Architecture Summary

**Purpose**: This program implements an entity-bridge multi-hop document retrieval system for the HOVER dataset that verifies factual claims by identifying named entities in retrieved documents and using bridge entities to discover connections between multi-hop facts.

**Key Modules**:
- **HoverMultiHopPipeline**: Core pipeline implementing entity-bridge retrieval architecture. Retrieves k=50 documents per hop (150 total), extracts named entities, identifies bridge entities that connect information, generates targeted queries, and applies LLM-based listwise reranking to select final 21 documents. Serves as the evaluation entry point.
- **ExtractEntities**: DSPy signature that extracts named entities (people, places, organizations, dates, events) from retrieved document passages with their types.
- **IdentifyBridgeEntities**: DSPy signature that analyzes claim and extracted entities to identify 3-5 bridge entities that likely connect to missing information, ranked by importance.
- **GenerateBridgedQuery**: DSPy signature that creates targeted search queries based on bridge entities and claim context to explore entity connections.
- **RerankDocuments**: DSPy signature that reranks all 150 retrieved documents to select top 21, prioritizing documents with multiple claim-relevant entities and entity connections.
- **hover_data.py**: Loads and preprocesses the HOVER dataset, filtering for 3-hop examples and formatting them as DSPy examples.
- **hover_utils.py**: Contains the evaluation metric `discrete_retrieval_eval` that checks if all gold supporting document titles are found within the retrieved documents (max 21).

**Data Flow**:
1. Input claim is passed to HoverMultiHopPipeline.forward()
2. Hop 1: Retrieve k=50 documents directly from claim, extract entities from passages
3. Identify bridge entities that likely connect to missing information (entities mentioned but not fully explored)
4. Hop 2: Generate targeted query for top bridge entity, retrieve k=50 more docs, extract entities
5. Combine entities and identify new bridge entities from accumulated information
6. Hop 3: Generate targeted query for next most promising bridge entity, retrieve k=50 final docs, extract entities
7. Combine all 150 retrieved documents and extracted entities
8. Apply LLM-based listwise reranking across all documents to select final 21, prioritizing documents containing multiple claim-relevant entities and entity connections
9. Return final 21 documents
10. Evaluation compares retrieved document titles against ground truth supporting_facts

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

