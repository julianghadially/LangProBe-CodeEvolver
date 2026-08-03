import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question based on the information we already have."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Answer the question directly and concisely, as an experienced StackExchange contributor would reply to another developer.

    Lead with the most practical, actionable guidance. Include concrete specifics (commands, names, settings, links) when they are the answer, but avoid padding with derived calculations or exhaustive detail the question did not ask for. Write natural prose without bracketed source citations such as [1] or [2]. Give a direct answer even when the context is incomplete rather than refusing or saying the context is insufficient."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    response = dspy.OutputField(desc="a direct, concise answer")
