import dspy


def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))


# Constraint: Do NOT return more than 21 documents for evaluation.
MAX_RETRIEVED_DOCS = 21


def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs[:MAX_RETRIEVED_DOCS]],
        )
    )
    return gold_titles.issubset(found_titles)
