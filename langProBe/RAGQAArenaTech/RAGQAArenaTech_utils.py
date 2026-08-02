import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a concise search query (a few keywords/phrases, not a full sentence) to find passages that answer the question.

    Keep the query CLOSE to the question's own key terms -- especially proper
    nouns, product names, error messages, and technical terms -- because those
    exact strings match the corpus best. You may add at most one or two
    clarifying terms from the context ONLY when they clearly disambiguate the
    question's intent; otherwise prefer the question's original wording.
    Do NOT drop key terms from the question, do NOT add speculative answer
    predictions, and do NOT narrow the query to a sub-topic the question did
    not ask about. The goal is broad recall of passages ABOUT the question's
    topic, not a narrow search for a pre-guessed answer.
    """

    context = dspy.InputField(desc="may contain relevant facts from prior searches")
    question = dspy.InputField()
    query = dspy.OutputField(desc="a concise keyword search query close to the question's terms")
