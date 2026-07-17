import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a focused search query to retrieve passages that answer the question from a large StackExchange-style technology corpus.

    First identify what the `context` already covers, then target the most important aspect of the question that the context does NOT yet answer.
    Use concrete technical terms likely to appear verbatim in the corpus (component names, product names, error codes, protocol identifiers, command/flag names, file extensions) rather than generic words.
    Output a short search-engine style query (keywords), NOT a full natural-language question."""

    context = dspy.InputField(desc="passages already retrieved in earlier hops; may already cover some aspects of the question")
    question = dspy.InputField()
    query = dspy.OutputField(desc="A concise search-engine style query of keywords and exact technical terms, targeting one not-yet-covered aspect of the question")
