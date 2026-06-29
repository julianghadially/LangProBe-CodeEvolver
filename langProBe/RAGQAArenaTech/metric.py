"""SemanticF1-based scoring + feedback metric for RAGQAArenaTech.

Mirrors the feedback-metric convention used by Hover/Hotpot
(langProPlus/hotpotGEPA/hotpot_metric_resource.py, gepa_optimize/programs/hover.py):

    ragqa_semantic_f1           -> float            (for dspy.Evaluate / simple_eval)
    ragqa_semantic_f1_feedback  -> ScoreWithFeedback (for GEPA / CodeEvolver reflection)

Scoring is identical to dspy.evaluate.SemanticF1 (the metric RAGQAArenaTech already
used): an LLM judge enumerates the key ideas in the gold and system responses and
reports recall + precision, combined as F1. The feedback variant additionally
surfaces the judge's reasoning and a directional hint so the optimizer learns *why*
a response scored low.
"""

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback
from dspy.evaluate.auto_evaluation import SemanticRecallPrecision, f1_score

# Judge LM, kept consistent with the original SemanticF1 wiring in
# langProBe/RAGQAArenaTech/__init__.py.
JUDGE_LM = "openai/gpt-5.4-mini"

# Matches dspy SemanticF1's default; only used in trace/bootstrap mode where the
# metric must return a bool.
THRESHOLD = 0.66

_judge = None


def _get_judge():
    """Lazily build the SemanticF1 judge (ChainOfThought) on its own LM."""
    global _judge
    if _judge is None:
        judge = dspy.ChainOfThought(SemanticRecallPrecision)
        judge.set_lm(dspy.LM(JUDGE_LM))
        _judge = judge
    return _judge


def _response_text(pred):
    """RAGQAArenaTech programs output a 'response' field; fall back gracefully."""
    return getattr(pred, "response", None) or getattr(pred, "answer", None) or str(pred)


def _judge_scores(question, gold_response, pred):
    """Run the judge; return (f1, scores) where scores has recall/precision/reasoning."""
    scores = _get_judge()(
        question=question,
        ground_truth=gold_response,
        system_response=_response_text(pred),
    )
    return f1_score(scores.precision, scores.recall), scores


def ragqa_semantic_f1(gold, pred, trace=None):
    """SemanticF1 score (float in [0, 1]); bool >= threshold under tracing/bootstrap."""
    score, _ = _judge_scores(gold.question, gold.response, pred)
    return score if trace is None else score >= THRESHOLD


def ragqa_semantic_f1_feedback(
    output, question, response, trace=None, pred_name=None, pred_trace=None
):
    """SemanticF1 score with textual feedback for GEPA / CodeEvolver reflection.

    Follows the CodeEvolver metric contract (see Hover's
    discrete_retrieval_eval_with_resource_penalty_and_feedback): the prediction is
    passed as ``output`` and the gold example's row fields are spread as keyword
    arguments (here ``question`` and ``response``).
    """
    pred = output  # rename
    gold_question = question
    gold_response = response
    score, scores = _judge_scores(gold_question, gold_response, pred)
    response = _response_text(pred)
    recall, precision = float(scores.recall), float(scores.precision)

    parts = [
        f"SemanticF1={score:.2f} (recall={recall:.2f}, precision={precision:.2f}).",
        f"Question: {gold_question}",
        f"Gold response: {gold_response}",
        f"System response: {response}",
    ]
    reasoning = getattr(scores, "reasoning", None)
    if reasoning:
        parts.append(f"Judge analysis: {reasoning}")

    # Directional guidance for the optimizer.
    if recall + precision == 0:
        parts.append(
            "The response shares no key ideas with the gold answer; reconsider both "
            "retrieval and answer generation."
        )
    elif recall < precision:
        parts.append(
            "Recall is the weaker side: the response omits key ideas present in the gold "
            "answer -- retrieve and cover more of the relevant content."
        )
    elif precision < recall:
        parts.append(
            "Precision is the weaker side: the response includes ideas not in the gold "
            "answer -- be more focused and faithful to the retrieved evidence."
        )

    return ScoreWithFeedback(score=score, feedback=" ".join(parts))
