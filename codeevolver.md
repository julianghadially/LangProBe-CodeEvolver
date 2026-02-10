PARENT_MODULE_PATH: langProBe.hover.hover_program.HoverMultiHopPredict
METRIC_MODULE_PATH: langProBe.hover.hover_utils.discrete_retrieval_eval

## Program Purpose
HoverMultiHopPredict is a multi-hop fact retrieval system designed for claim verification on the HoVer benchmark. It performs iterative retrieval across Wikipedia documents to find all supporting facts (gold documents) needed to verify a given claim, using up to 3 retrieval hops with query generation and document summarization.

## Key Modules and Responsibilities

**HoverMultiHopPredict** (hover_program.py): The main DSPy module that orchestrates the 3-hop retrieval pipeline. Contains:
- `retrieve_k`: Retrieves top-k (k=7) documents per hop using DSPy's Retrieve module
- `create_query_hop2` and `create_query_hop3`: Generate refined search queries for subsequent hops based on claim and previous summaries
- `summarize1` and `summarize2`: Condense retrieved passages into summaries to inform next hop queries

**hoverBench** (hover_data.py): Benchmark data loader that:
- Loads the HoVer dataset from Hugging Face (hover-nlp/hover)
- Filters examples to include only 3-hop fact verification tasks (count_unique_docs == 3 for training, ≤3 for test)
- Formats data as DSPy Examples with "claim" as input field

**discrete_retrieval_eval** (hover_utils.py): Evaluation metric that:
- Compares retrieved document titles against gold supporting facts
- Returns True if all gold document titles are present in retrieved documents (subset check)
- Enforces MAX_RETRIEVED_DOCS=21 constraint

## Data Flow
1. Input claim → Hop 1: retrieve_k(claim) → summarize top-7 documents
2. Hop 2: generate query from (claim, summary_1) → retrieve_k → summarize with context
3. Hop 3: generate query from (claim, summary_1, summary_2) → retrieve_k
4. Output: Concatenated list of all retrieved documents (21 max) from 3 hops
5. Evaluation: Check if all gold supporting facts are in retrieved set

## Metric Being Optimized
The `discrete_retrieval_eval` metric measures retrieval recall: whether the system successfully retrieved ALL documents containing supporting facts for the claim. It returns a binary score (1 if all gold documents found, 0 otherwise), evaluating the system's ability to gather complete evidence through multi-hop reasoning.

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

