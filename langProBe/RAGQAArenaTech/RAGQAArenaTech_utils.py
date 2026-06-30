import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question based on the information we already have."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateDiverseSearchQueries(dspy.Signature):
    """Write up to 3 diverse search queries that, together, will help answer a complex question.

    Each query should target a different aspect/entity/surface form of the question
    (e.g. rephrasings, named-entity-focused variants, or platform-specific wording)
    so that retrieval has multiple angles to find the gold passages. Avoid near
    duplicates -- make the queries genuinely different.
    """

    context = dspy.InputField(desc="may contain relevant facts already gathered")
    question = dspy.InputField()
    queries = dspy.OutputField(
        desc="a newline-separated list of up to 3 diverse search queries"
    )


class AssessPassageRelevance(dspy.Signature):
    """Judge whether a retrieval passage is relevant to answering the user's question.

    Score 0 = irrelevant/off-topic/noise; 1 = marginally relevant; 2 = clearly relevant and useful.
    Be strict: a passage that only shares vocabulary or is tangential should score 0 or 1, not 2.
    """

    question = dspy.InputField()
    passage = dspy.InputField(desc="a retrieved text passage")
    score = dspy.OutputField(desc="integer relevance score: 0, 1, or 2")