import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a simple, plain-text search query (a few keywords) that will help answer a complex question based on the information we already have. Output only the query string; never output shell syntax, braces, or code snippets."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField(desc="a short plain-text search query of a few keywords")


class GenerateAnswer(dspy.Signature):
    """Answer the user's question using only the retrieved context.

    - Ground every factual claim in the retrieved passages; do not state anything the passages do not support. Untruthful or unsupported content is penalized first.
    - Be complete: cover all relevant points the question asks for that the passages support.
    - Be concise and focused: write a direct prose answer in a few sentences. Do not pad with tangential detail, and do not provide unsafe, harmful, or offensive guidance.
    - Address the question's actual intent; if it is ambiguous, answer the most likely intended meaning rather than a generic one.

    Write the final answer as plain prose in the `response` field. Always fill in the response field; never leave it empty.
    """

    context = dspy.InputField(desc="retrieved passages containing facts relevant to the question")
    question = dspy.InputField()
    response = dspy.OutputField(desc="a concise, complete prose answer grounded only in the retrieved passages")
