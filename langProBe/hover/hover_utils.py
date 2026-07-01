import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback


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


# ---------------------------------------------------------------------------
# Resource-penalty metric (with textual feedback for reflective optimization)
# ---------------------------------------------------------------------------
#
# Score = retrieval success (all gold docs in top-21) penalized by the number of
# search queries used. The search count is read from ``pred.search_count``, which
# the pipeline attaches via its thread-safe CountingRM wrapper.
#
# There is NO hard cap on the number of searches -- using more searches is fully
# allowed when it is worthwhile. The penalty below is purely a soft resource-use
# cost so the optimizer trades off retrieval effort against accuracy.
#
# The only hard constraint (see specs/requirements.md) is on the final output:
#   - Do NOT return more than 21 documents.

PENALTY_PER_SEARCH = 0.002
# Searches up to this budget are "free"; each additional search is penalized.
FREE_SEARCH_BUDGET = 2


def calculate_search_penalty(search_count):
    return PENALTY_PER_SEARCH * max(0, search_count - FREE_SEARCH_BUDGET)


def discrete_retrieval_eval_with_resource_penalty(example, pred, trace=None):
    """Retrieval success penalized by search count. Returns a float."""
    success = float(discrete_retrieval_eval(example, pred, trace))
    search_count = getattr(pred, "search_count", 0)
    penalty = calculate_search_penalty(search_count)
    return max(0.0, success - penalty)

def _example_field(example, name):
    """Read a field from the example row (dspy.Example attr or plain mapping)."""
    if hasattr(example, name):
        return getattr(example, name)
    try:
        return example[name]
    except (TypeError, KeyError):
        return None

def discrete_retrieval_eval_with_resource_penalty_and_feedback(
    output, example, trace=None, pred_name=None, pred_trace=None
):
    """Retrieval success penalized by search count, with feedback for reflection."""
    pred = output  # rename
    # The full dataset row arrives as `example` (attr + item access); gold
    # fields are read off of it.
    supporting_facts = _example_field(example, "supporting_facts")
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in supporting_facts],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs[:MAX_RETRIEVED_DOCS]],
        )
    )
    success = float(gold_titles.issubset(found_titles))
    missing = gold_titles - found_titles

    search_count = getattr(pred, "search_count", 0)
    penalty = calculate_search_penalty(search_count)
    composite = max(0.0, success - penalty)

    feedback = (
        "SUCCESS: all supporting documents retrieved. "
        if success
        else f"INCOMPLETE: {len(missing)}/{len(gold_titles)} supporting documents missing. "
    )
    feedback += (
        f"Required supporting documents: {sorted(gold_titles)}. "
    )
    if missing:
        feedback += f"Missing documents: {sorted(missing)}. "
    feedback += (
        f"Documents retrieved: {len(pred.retrieved_docs[:MAX_RETRIEVED_DOCS])} "
        f"(hard cap: {MAX_RETRIEVED_DOCS}). "
        f"Searches used: {search_count} (Searching more than "
        f"{FREE_SEARCH_BUDGET} costs a small penalty of {PENALTY_PER_SEARCH:g}), "
        f"penalty: {penalty:g}, composite score: {composite:g}."
    )

    return ScoreWithFeedback(score=composite, feedback=feedback)
