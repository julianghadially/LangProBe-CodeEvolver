import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a search query that will help answer a complex question. If the question uses everyday or colloquial wording that might refer to a specific device feature, setting, icon, or symptom the user is seeing, infer the most likely concrete intent behind the wording and write a query targeting that specific feature rather than a literal interpretation of the words. If the question starts with 'how to', 'how do I', or 'how can I', and a literal reading would retrieve general background or encyclopedic info, target the method or procedure, not the topic. If the context already contains relevant facts, target information NOT already covered -- alternate facets, a different aspect of the question, or missing details -- rather than re-searching what you already have."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Answer the question directly and concisely, as an experienced StackExchange contributor would reply to another developer.

    Lead with the most practical, actionable guidance. If the question has several distinct aspects or approaches, briefly address each one rather than only the most actionable. Include specific protocol types from the context when they are directly relevant to the answer. When the question asks for a specific quantity, count, or measurement and the context provides exact numbers or estimates, include those values in your answer. Avoid padding with derived calculations or exhaustive detail the question did not ask for. Write natural prose without bracketed source citations such as [1] or [2]. Give a direct answer even when the context is incomplete rather than refusing or saying the context is insufficient."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    response = dspy.OutputField(desc="a direct, concise answer")
