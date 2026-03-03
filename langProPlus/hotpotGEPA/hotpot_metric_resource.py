"""Composite accuracy with retrieval resource penalty metric for HotpotQA.

Measures answer correctness (exact match) and applies a penalty of 0.02
for each retrieval query performed.  The retrieval count is read from
``pred.retrieval_count``, which is attached automatically by the pipeline's
CountingRM wrapper — no manual tracking required.

Variants:
    hotpot_accuracy_with_resource_penalty          — returns a float (for Evaluate)
    hotpot_accuracy_with_resource_penalty_feedback  — returns ScoreWithFeedback (for GEPA)
    hotpot_llm_judge                               — exact match then LLM judge fallback
    hotpot_llm_judge_feedback                      — same with ScoreWithFeedback (for GEPA)
"""

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

PENALTY_PER_RETRIEVAL = 0.02

def calculate_penalty(retrieval_count):
    return PENALTY_PER_RETRIEVAL * max(0.0, retrieval_count - 2)


def hotpot_accuracy_with_resource_penalty(gold, pred, trace=None):
    """Accuracy penalized by retrieval count (0.02 per query)."""
    accuracy = float(dspy.evaluate.answer_exact_match(gold, pred))
    retrieval_count = getattr(pred, "retrieval_count", 0)
    penalty = calculate_penalty(retrieval_count)
    return max(0.0, accuracy - penalty)


def hotpot_accuracy_with_resource_penalty_feedback(
    gold, pred, trace=None, pred_name=None, pred_trace=None
):
    """Accuracy penalized by retrieval count, with textual feedback for GEPA."""
    accuracy = float(dspy.evaluate.answer_exact_match(gold, pred))
    retrieval_count = getattr(pred, "retrieval_count", 0)
    penalty = calculate_penalty(retrieval_count)
    composite = max(0.0, accuracy - penalty)

    feedback = f"Gold answer: '{gold.answer}'. Predicted: '{pred.answer}'. "
    feedback += "CORRECT. " if accuracy else "INCORRECT. "
    feedback += (
        f"Queries used: {retrieval_count}, "
        f"Penalty: {penalty:.2f} (Using more queries may improve accuracy but carries a small penalty), "
        f"composite: {composite:.2f}. "
    )
    gold_titles = gold.get("gold_titles", [])
    if gold_titles:
        feedback += f"Supporting documents needed: {gold_titles}."

    return ScoreWithFeedback(score=composite, feedback=feedback)


# ---------------------------------------------------------------------------
# LLM-as-a-judge metric
# ---------------------------------------------------------------------------

class AnswerEquivalenceJudge(dspy.Signature):
    """Judge whether a predicted answer is semantically equivalent to the gold answer."""
    gold_answer: str = dspy.InputField(desc="The correct reference answer")
    predicted_answer: str = dspy.InputField(desc="The model's predicted answer")
    equivalent: bool = dspy.OutputField(desc="True if the predicted answer is semantically equivalent to the gold answer, False otherwise")


def _exact_match_normalized(gold: str, pred: str) -> bool:
    """Case-insensitive, whitespace-stripped exact match."""
    return gold.lower().strip() == pred.lower().strip()


def hotpot_llm_judge(gold, pred, trace=None):
    """Exact string match (normalized), then LLM judge fallback. Returns float."""
    gold_answer = gold.answer
    pred_answer = getattr(pred, "answer", str(pred))
    string_match_score = _exact_match_normalized(gold_answer, pred_answer)
    if string_match_score > 0.9:
        return 1.0

    try:
        judge = dspy.Predict(AnswerEquivalenceJudge)
        result = judge(gold_answer=gold_answer, predicted_answer=pred_answer)
        return 1.0 if result.equivalent else 0.0
    except Exception as e:
        print(f"WARNING: LLM judge failed, falling back to string match score: {e}")
        return string_match_score


def hotpot_llm_judge_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Exact string match (normalized), then LLM judge fallback. Returns ScoreWithFeedback."""
    gold_answer = gold.answer
    pred_answer = getattr(pred, "answer", str(pred))

    score = hotpot_llm_judge(gold, pred)

    feedback = f"Gold answer: '{gold_answer}'. Predicted: '{pred_answer}'. "
    feedback += "CORRECT" if score else "INCORRECT"

    return ScoreWithFeedback(score=score, feedback=feedback)
