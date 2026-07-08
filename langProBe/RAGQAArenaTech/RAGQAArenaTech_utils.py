import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a simple, plain-text search query (a few keywords) that will help answer a complex question based on the information we already have. Output only the query string; never output shell syntax, braces, or code snippets."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField(desc="a short plain-text search query of a few keywords")


class GenerateAnswer(dspy.Signature):
    """Answer the user's question using the retrieved context to ground your response.

    - The retrieved passages are your primary source of evidence. State the relevant facts, methods, options, or steps they describe for the question.
    - Be complete and thorough: enumerate all relevant points, alternatives, or steps the passages contain that answer the question. The expected answer is a thorough long-form answer that covers the available evidence, not a one-liner.
    - Do not refuse to answer. If the retrieved context only partially addresses the question, answer the supported parts fully and give your best helpful answer for the rest rather than saying you cannot answer.
    - Address the question's actual intent; if it is ambiguous, answer the most likely intended meaning.
    - Do not prefix the answer with disclaimers about what the context lacks (e.g., "The retrieved context does not provide..."). If the passages do not directly answer the question, lead with the closest helpful content they do support.
    - Do not append speculative caveats, hedging qualifiers, or unsolicited "you may also want to..." / "alternatively, you could..." suggestions that are not grounded in the retrieved passages. State what the passages support, then stop.
    - Do not echo passage index markers (e.g., [1], [10]) from the retrieved passages into the response.
    - Do not invent specific false facts, commands, or figures not supported by the passages, and do not provide unsafe, harmful, or offensive guidance.

    Write the final answer as plain prose (short lists or code snippets are fine when the passages contain them) in the `response` field. Always fill in the response field; never leave it empty.
    """

    context = dspy.InputField(desc="retrieved passages containing facts relevant to the question")
    question = dspy.InputField()
    response = dspy.OutputField(desc="a thorough, well-grounded prose answer to the question")
