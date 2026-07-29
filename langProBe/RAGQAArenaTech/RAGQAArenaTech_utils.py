import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a concise search query that will retrieve passages answering the question.

    Preserve the question's specific terms (product names, UI element names,
    error messages, domain vocabulary) so retrieval targets the intended feature or
    problem rather than a generic concept. Drop filler words but keep the nouns that
    disambiguate the user's actual intent.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
