"""Composite accuracy with retrieval resource penalty metric for HotpotQA.

Measures answer correctness (exact match) and applies a penalty of 0.02
for each retrieval query performed.  The retrieval count is read from
``pred.retrieval_count``, which is attached automatically by the pipeline's
CountingRM wrapper — no manual tracking required.

Two variants:
    hotpot_accuracy_with_resource_penalty          — returns a float (for Evaluate)
    hotpot_accuracy_with_resource_penalty_feedback  — returns ScoreWithFeedback (for GEPA)
"""

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

PENALTY_PER_RETRIEVAL = 0.0025

def calculate_penalty(retrieval_count):
    return PENALTY_PER_RETRIEVAL * max(0.0, retrieval_count - 2)


def hotpot_accuracy_with_resource_penalty(gold, pred, trace=None):
    """Accuracy penalized by retrieval count (0.02 per query)."""
    accuracy = float(dspy.evaluate.answer_exact_match(gold, pred))
    retrieval_count = getattr(pred, "retrieval_count", 0)
    penalty = calculate_penalty(retrieval_count)
    return max(0.0, accuracy - penalty)


def hotpot_accuracy_with_resource_penalty_feedback(
    output=None, answer=None, gold_titles=None, trace=None, pred_name=None, pred_trace=None
):
    """Accuracy penalized by retrieval count, with textual feedback for GEPA."""
    pred = output
    gold = answer
    accuracy = float(dspy.evaluate.answer_exact_match(gold, pred))
    retrieval_count = getattr(pred, "retrieval_count", 0)
    penalty = calculate_penalty(retrieval_count)
    composite = max(0.0, accuracy - penalty)

    feedback = f"Gold answer: '{gold}'. Predicted: '{pred.answer}'. "
    feedback += "CORRECT. " if accuracy else "INCORRECT. "
    feedback += (
        f"Queries used: {retrieval_count}, "
        f"Penalty: {penalty:.2f} (Using more queries may improve accuracy but carries a small penalty), "
        f"composite: {composite:.2f}. "
    )
    if gold_titles:
        feedback += f"Supporting documents needed: {gold_titles}."

    return ScoreWithFeedback(score=composite, feedback=feedback)
