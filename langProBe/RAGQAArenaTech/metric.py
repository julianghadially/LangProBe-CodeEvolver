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

import json
import os
import re

import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback
from dspy.evaluate.auto_evaluation import (
    AnswerCompleteness,
    AnswerGroundedness,
    SemanticRecallPrecision,
    f1_score,
)

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


def _example_field(example, name):
    """Read a field from the example row (dspy.Example attr or plain mapping)."""
    if hasattr(example, name):
        return getattr(example, name)
    try:
        return example[name]
    except (TypeError, KeyError):
        return None


def ragqa_semantic_f1_feedback(
    output, example, trace=None, pred_name=None, pred_trace=None
):
    """SemanticF1 score with textual feedback for GEPA / CodeEvolver reflection.

    Follows the CodeEvolver metric contract: the prediction is passed as ``output``
    and the entire gold example row is passed as ``example`` (a dspy.Example), so
    the gold fields are read off it -- ``example.question`` and ``example.response``.
    Settings/config kwargs (``trace``, ``pred_name``, ``pred_trace``) stay separate.
    """
    pred = output  # rename
    gold_question = _example_field(example, "question")
    gold_response = _example_field(example, "response")
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



# ======================================================================================
# PAIRWISE LFRQA WIN-RATE -- RAG-QA Arena's own metric (arXiv:2407.13998)
# ======================================================================================
#
# Faithful reproduction of the model-based pairwise evaluation from the official repo
# (github.com/awslabs/rag-qa-arena): `code/data_processors.py::LFRQADataProcessor`,
# `code/compute_correlation.py`, `code/report_results.py`, and the verbatim prompt
# assets in `templates/` (vendored under ./rqa_arena_eval/).
#
# Protocol (exactly as the paper):
#   1. The judge sees a QUERY and TWO answers (the system answer vs. the human LFRQA
#      "faithful_answer"). It sees NO retrieved passages (the LFRQA-comparison template
#      has no passages field -- faithful).
#   2. The system prompt carries the truthfulness/helpfulness rubric
#      (rqa_arena_eval/pairwise_lfrqa_system.txt, verbatim); three in-context examples
#      (rqa_arena_eval/pairwise_lfrqa_examples.json, verbatim, labels 1/2/0) are
#      injected as user/assistant turns; the final user turn is the rendered template
#      (rqa_arena_eval/pairwise template below, verbatim from pairwise_lfrqa.cfg).
#   3. Position bias: the paper assigns the system answer to slot 1 or slot 2
#      *deterministically by the parity of the query word count*
#      (`len(query.split(' ')) % 2 == 0`), and judges each query EXACTLY ONCE. We
#      replicate this exactly (see `_pairwise_slot_order`). It is NOT a bidirectional
#      both-orders average.
#   4. The judge outputs `<rating>0|1|2</rating>`: 1 => prefer answer 1, 2 => prefer
#      answer 2, 0 => not sure (tie). We map the rating back through the slot order to
#      a win / tie / loss for the SYSTEM answer.
#   5. The paper reports two dataset-level numbers: W (win rate) and W+T (win+tie).
#
# ------------------------------------------------------------------------------------
# DEVIATIONS FROM THE PAPER (all called out, per request):
#
#   [D1] Single scalar per example. DSPy/CodeEvolver metrics must return ONE float per
#        example, but the paper reports W and W+T as two separate aggregates. We map
#        win=1.0 / tie=0.5 / loss=0.0, so the dataset MEAN of this metric equals
#        `W + 0.5*T` (the midpoint of W and W+T). To recover W and W+T separately, use
#        `ragqa_pairwise_outcome(...)` which returns the raw 'win'/'tie'/'loss' string.
#
#   [D2] Bool/trace mode. `dspy.Evaluate` bootstrap/trace mode needs a bool; we treat a
#        WIN (score == 1.0) as the positive label (`>= PAIRWISE_THRESHOLD`). This is a
#        DSPy-framework requirement, absent from the paper.
# ------------------------------------------------------------------------------------

# [D2] Paper's model-based evaluator.
PAIRWISE_JUDGE_LM = "openai/gpt-5.4-mini"
# [D4] Deterministic judging.
PAIRWISE_JUDGE_TEMPERATURE = 0.0
# [D3] 1 == single evaluator == paper's headline setup. >1 draws N votes + majority.
PAIRWISE_JUDGE_SAMPLES = 1
# [D6] A "win" (1.0) is the positive label under bootstrap/trace mode.
PAIRWISE_THRESHOLD = 1.0

# From templates/pairwise_lfrqa.cfg (`{x.question}` etc. -> str.format fields). The
# query/answer/rubric structure is verbatim; [D7] the final instruction asks for a
# visible one-line <reason> (instead of the paper's <thinking>) so the judge's
# rationale lands in the VISIBLE text and can be surfaced to the CodeEvolver optimizer
# -- reasoning-model judges route <thinking>-style deliberation to a separate
# reasoning_content channel, leaving the paper's <thinking> tag empty in the output.
_PAIRWISE_TEMPLATE = """Query is in the <query></query> tags. Answer 1 is in <answer 1></answer 1>, and Answer 2 is in <answer 2></answer 2>.

<query>
{question}
</query>

<answer 1>
{response1}
</answer 1>

<answer 2>
{response2}
</answer 2>

Review the rubric in <rubric> tags,
- if you prefer <answer 1>, output 1.
- if you prefer <answer 2>, output 2.
- if you are not sure, output 0.

First, in one sentence (shorter than 50 words), state the single most important reason for your preference inside <reason></reason> tags. Then, provide your rating inside <rating></rating> tags. Remember your rating should be 0 if you are not sure, and your rating must be either 0, 1, or 2. """

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "rqa_arena_eval")
_RATING_TAG_RE = re.compile(r"<rating>.*?</rating>", re.DOTALL)

_pairwise_judge = None
_pairwise_system = None
_pairwise_examples = None


def _get_pairwise_judge():
    """Lazily build the pairwise judge LM ([D2]/[D4])."""
    global _pairwise_judge
    if _pairwise_judge is None:
        _pairwise_judge = dspy.LM(
            PAIRWISE_JUDGE_LM, temperature=PAIRWISE_JUDGE_TEMPERATURE
        )
    return _pairwise_judge


def _load_pairwise_assets():
    """Load the verbatim system prompt + 3 in-context examples once."""
    global _pairwise_system, _pairwise_examples
    if _pairwise_system is None:
        with open(os.path.join(_ASSET_DIR, "pairwise_lfrqa_system.txt")) as f:
            _pairwise_system = f.read()
        with open(os.path.join(_ASSET_DIR, "pairwise_lfrqa_examples.json")) as f:
            _pairwise_examples = json.load(f)
    return _pairwise_system, _pairwise_examples


def _render_pairwise(question, response1, response2):
    return _PAIRWISE_TEMPLATE.format(
        question=question, response1=response1, response2=response2
    )


# Faithful port of utils.py::process_response (strips thinking/answer scaffolding).
def _process_response(response):
    s = str(response)
    for open_t, close_t in (
        ("<thinking>", "</thinking>"),
        ("Thinking: ", "Answer: "),
        ("Thoughts: ", "Answer: "),
        ("Thought: ", "Answer: "),
    ):
        i, j = s.find(open_t), s.find(close_t)
        if 0 <= i <= j:
            s = s[j + len(close_t):]
    s = s.strip()
    for tag in ("Answer: ", "Answer:\n", "<answer>", "</answer>"):
        s = s.replace(tag, "")
    if s in ("FAIL TO GENERATE ANS.",):
        return "I couldn't find an answer."
    return s


def _pairwise_slot_order(question, system_answer, gold_answer):
    """Paper's deterministic position assignment ([position bias], step 3).

    Even query word count -> system answer is slot 1, LFRQA is slot 2.
    Odd  query word count -> LFRQA is slot 1, system answer is slot 2.
    Returns (response1, response2, system_slot) where system_slot in {1, 2}.
    """
    if len(question.split(" ")) % 2 == 0:
        return system_answer, gold_answer, 1
    return gold_answer, system_answer, 2


def _parse_rating(completion):
    """Extract the judge's rating in {0, 1, 2} from `<rating>..</rating>` ([D5]).

    Mirrors compute_correlation.py: take the first `<rating>` match, then the highest
    of {0,1,2} whose digit appears inside it. Missing tag -> None (caller: tie)."""
    m = _RATING_TAG_RE.findall(str(completion))
    if not m:
        return None
    tag = m[0]
    rating = None
    for i in (0, 1, 2):
        if str(i) in tag:
            rating = i
    return rating


def _judge_once(question, response1, response2):
    """One judge call over the assembled (system + 3 ICL turns + final) messages."""
    system, examples = _load_pairwise_assets()
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append(
            {"role": "user",
             "content": _render_pairwise(ex["query"], ex["response_1"], ex["response_2"])}
        )
        messages.append(
            {"role": "assistant",
             "content": f"<reason>{ex['thinking']}</reason><rating>{ex['label']}</rating>"}
        )
    messages.append({"role": "user", "content": _render_pairwise(question, response1, response2)})
    out = _get_pairwise_judge()(messages=messages)
    return out[0] if isinstance(out, list) else str(out)


def _pairwise_judge_call(question, gold_response, system_response):
    """Run the paper's pairwise comparison for one row.

    Returns (outcome, score, rating, completion) where outcome in
    {'win','tie','loss'} for the SYSTEM answer and score in {1.0, 0.5, 0.0} ([D1])."""
    pred = _process_response(system_response)
    reference = _process_response(gold_response)
    response1, response2, system_slot = _pairwise_slot_order(question, pred, reference)

    # [D3] one vote (paper's headline), or N-sample majority if configured.
    votes, last_completion = [], ""
    for _ in range(max(1, PAIRWISE_JUDGE_SAMPLES)):
        try:
            last_completion = _judge_once(question, response1, response2)
        except Exception:
            last_completion = ""  # [D5] judge failure -> tie
        votes.append(_parse_rating(last_completion))

    rating = _majority_rating(votes)  # None on tie/no-majority/unparseable
    if rating is None or rating == 0:
        return "tie", 0.5, (rating if rating is not None else 0), last_completion
    if rating == system_slot:
        return "win", 1.0, rating, last_completion
    return "loss", 0.0, rating, last_completion


def _majority_rating(votes):
    """[D3] Majority over judge votes; None if tie/no-majority (-> caller treats as tie).

    With a single vote (paper's default) this returns that vote unchanged (a bare
    None stays None -> tie)."""
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    from collections import Counter
    counts = Counter(valid).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None  # no majority -> tie
    return counts[0][0]


def ragqa_pairwise_outcome(gold, pred, trace=None):
    """Raw pairwise outcome 'win'/'tie'/'loss' for the system answer vs. LFRQA.

    Exposed so callers can compute the paper's W and W+T separately ([D1]):
        W   = mean(outcome == 'win')
        W+T = mean(outcome in {'win','tie'})
    """
    outcome, _, _, _ = _pairwise_judge_call(
        gold.question, gold.response, _response_text(pred)
    )
    return outcome


def ragqa_pairwise_winrate(gold, pred, trace=None):
    """Pairwise win-rate score (win=1.0 / tie=0.5 / loss=0.0); bool WIN under tracing."""
    _, score, _, _ = _pairwise_judge_call(
        gold.question, gold.response, _response_text(pred)
    )
    return score if trace is None else score >= PAIRWISE_THRESHOLD


def _extract_thinking(completion):
    m = re.search(r"<thinking>(.*?)</thinking>", str(completion), re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_reason(completion):
    """The judge's one-line preference reason, for optimizer feedback ([D7]).

    Primary: the visible ``<reason>..</reason>`` tag the judge is now asked to emit.
    Fallback: the reasoning model's native ``reasoning_content`` channel (a reasoning
    judge may route its rationale there and emit only ``<rating>`` in the visible
    text). Whitespace/newlines are collapsed to keep it one line. '' if neither exists.
    """
    text = completion.get("text", "") if isinstance(completion, dict) else str(completion)
    m = re.search(r"<reason>(.*?)</reason>", str(text), re.DOTALL)
    if m and m.group(1).strip():
        return " ".join(m.group(1).split())
    if isinstance(completion, dict):
        rc = (completion.get("reasoning_content") or "").strip()
        if rc:
            return " ".join(rc.split())
    m2 = re.search(r"<reason>(.*?)</reason>", str(completion), re.DOTALL)
    return " ".join(m2.group(1).split()) if m2 else ""


def ragqa_pairwise_winrate_feedback(
    output, example, trace=None, pred_name=None, pred_trace=None
):
    """Pairwise win-rate with textual feedback for GEPA / CodeEvolver reflection.

    Same CodeEvolver metric contract as ``ragqa_semantic_f1_feedback``: the prediction
    is ``output`` and the gold example row is ``example`` (fields ``question`` and
    ``response`` -- the LFRQA faithful answer)."""
    pred = output
    gold_question = _example_field(example, "question")
    gold_response = _example_field(example, "response")
    response = _response_text(pred)
    outcome, score, rating, completion = _pairwise_judge_call(
        gold_question, gold_response, response
    )

    parts = [
        f"Pairwise vs LFRQA: {outcome.upper()} (score={score:.1f}, judge rating={rating}).",
        f"Question: {gold_question}",
        f"Gold (LFRQA) answer: {gold_response}",
        f"System response: {response}",
    ]
    reason = _extract_reason(completion)
    if reason:
        parts.append(f"Judge reason: {reason}")

    # Directional guidance for the optimizer (mirrors the SemanticF1 feedback style).
    if outcome == "loss":
        parts.append(
            "The judge preferred the human answer -- it was more truthful/helpful/"
            "complete. Improve factual grounding and coverage of the query without "
            "adding unsupported claims (untruthful content is penalized first)."
        )
    elif outcome == "tie":
        parts.append(
            "Judged a tie. To WIN, add truthful, query-relevant coverage the human "
            "answer lacks -- but keep every claim grounded, since any untruthful "
            "information is penalized ahead of completeness."
        )
    else:
        parts.append(
            "The system answer beat the human reference (more truthful/helpful/"
            "complete). Preserve this grounding and coverage."
        )

    return ScoreWithFeedback(score=score, feedback=" ".join(parts))



# ======================================================================================
# COMPLETENESS / FAITHFULNESS / CONCISENESS (CFC)
# ======================================================================================
#
# A composite metric with more optimization headroom than the (saturated) LFRQA pairwise
# win-rate. The total is a weighted sum of three components, each in [0, 1]:
#
#   total = 0.50 * completeness   -- fraction of the GROUND-TRUTH answer covered by the
#                                    response (are we saying everything we should?).
#         + 0.25 * faithfulness   -- fraction of the response SUPPORTED BY the retrieved
#                                    passages (are our claims grounded in evidence?).
#         + 0.25 * conciseness    -- a length penalty that punishes over-long answers.
#
#   * Completeness   -> DSPy's AnswerCompleteness (auto_evaluation.py): LLM judge that
#     enumerates key ideas in gold vs. response and reports `completeness`.
#   * Faithfulness   -> DSPy's AnswerGroundedness (auto_evaluation.py): LLM judge that
#     enumerates the response's check-worthy claims and reports the fraction `groundedness`
#     deducible from the retrieved context. Needs the retrieved passages -- SimplifiedBaleen
#     now carries them on `pred.context` (see RAGQAArenaTech_program.py).
#   * Conciseness    -> pure-Python char-length penalty; NO LLM call (deterministic, free).
#
# Both judges reuse JUDGE_LM ("openai/gpt-5.4-mini"), consistent with the metrics above.

CFC_WEIGHTS = {"completeness": 0.5, "faithfulness": 0.25, "conciseness": 0.25}

# Conciseness "line with a cliff" knees (ratio = len(response) / len(gold), in chars):
#   ratio <= 1.3          -> no penalty              (the first 30% over gold is exempt)
#   1.3 < ratio < 3.0     -> penalty = (ratio-1)/2   (a line anchored at (1.0, 0), so it
#                                                     jumps to 0.15 at 1.3+ -- the cliff)
#   ratio >= 3.0          -> full penalty (1.0)
CONCISENESS_EXEMPT_RATIO = 1.3
CONCISENESS_FULL_RATIO = 3.0

# A "good" composite answer under bootstrap/trace mode (matches the file's 0.66 convention).
CFC_THRESHOLD = 0.66

_completeness_judge = None
_groundedness_judge = None


def _get_completeness_judge():
    """Lazily build the DSPy AnswerCompleteness judge on JUDGE_LM."""
    global _completeness_judge
    if _completeness_judge is None:
        judge = dspy.ChainOfThought(AnswerCompleteness)
        judge.set_lm(dspy.LM(JUDGE_LM))
        _completeness_judge = judge
    return _completeness_judge


def _get_groundedness_judge():
    """Lazily build the DSPy AnswerGroundedness (faithfulness) judge on JUDGE_LM."""
    global _groundedness_judge
    if _groundedness_judge is None:
        judge = dspy.ChainOfThought(AnswerGroundedness)
        judge.set_lm(dspy.LM(JUDGE_LM))
        _groundedness_judge = judge
    return _groundedness_judge


def _context_text(pred, pred_trace=None):
    """Return the retrieved passages the answer was generated from, as one string.

    Primary source: ``pred.context`` (a list[str]), which SimplifiedBaleen attaches to
    the returned prediction. Fallback: recover the ``respond`` predictor's ``context``
    input from ``pred_trace``. Returns "" if unavailable (an evolved program may have
    dropped it) -- the caller then scores faithfulness as 0.0 and flags it in feedback.
    """
    ctx = getattr(pred, "context", None)
    if not ctx and pred_trace:
        # pred_trace is a list of (predictor, inputs, outputs); the answer generator's
        # inputs carry the passage list under "context".
        for step in pred_trace:
            try:
                inputs = step[1]
            except (TypeError, IndexError):
                continue
            if isinstance(inputs, dict) and inputs.get("context"):
                ctx = inputs["context"]
                break
    if not ctx:
        return ""
    if isinstance(ctx, (list, tuple)):
        return "\n\n".join(str(p) for p in ctx)
    return str(ctx)


def _conciseness_score(response, gold_response):
    """Length-penalty conciseness score in [0, 1]: a straight line with a cliff.

    Penalizes responses that run long relative to the gold answer's character count.
    See CONCISENESS_* knees above for the exact shape. Being *shorter* than gold is
    never penalized here (under-coverage is already handled by completeness).
    """
    gold_len = len(gold_response or "")
    if gold_len == 0:
        return 1.0  # undefined ratio -> don't penalize
    ratio = len(response or "") / gold_len
    if ratio <= CONCISENESS_EXEMPT_RATIO:
        penalty = 0.0
    elif ratio >= CONCISENESS_FULL_RATIO:
        penalty = 1.0
    else:
        penalty = (ratio - 1.0) / (CONCISENESS_FULL_RATIO - 1.0)
    return max(0.0, min(1.0, 1.0 - penalty))


def _cfc_components(question, gold_response, response, context_text):
    """Compute (completeness, faithfulness, conciseness) plus judge discussions.

    Returns (comp, faith, concise, meta) where meta carries the judges' short
    discussion text and the char-length ratio for feedback."""
    comp_out = _get_completeness_judge()(
        question=question, ground_truth=gold_response, system_response=response
    )
    completeness = max(0.0, min(1.0, float(comp_out.completeness)))

    faith_grounded = False
    if context_text:
        faith_grounded = True
        ground_out = _get_groundedness_judge()(
            question=question,
            retrieved_context=context_text,
            system_response=response,
        )
        faithfulness = max(0.0, min(1.0, float(ground_out.groundedness)))
        ground_discussion = getattr(ground_out, "discussion", "")
    else:
        # Retrieval evidence not exposed on the prediction -> a RAG answer with no
        # grounding evidence cannot be judged faithful. Flag it so the optimizer keeps
        # carrying the retrieved context.
        faithfulness = 0.0
        ground_discussion = (
            "No retrieved context was exposed on the prediction (pred.context missing), "
            "so faithfulness could not be assessed and is scored 0.0."
        )

    conciseness = _conciseness_score(response, gold_response)

    gold_len = len(gold_response or "")
    ratio = (len(response or "") / gold_len) if gold_len else 0.0
    meta = {
        "completeness_discussion": getattr(comp_out, "discussion", ""),
        "groundedness_discussion": ground_discussion,
        "length_ratio": ratio,
        "faith_grounded": faith_grounded,
    }
    return completeness, faithfulness, conciseness, meta


def _cfc_total(completeness, faithfulness, conciseness):
    return (
        CFC_WEIGHTS["completeness"] * completeness
        + CFC_WEIGHTS["faithfulness"] * faithfulness
        + CFC_WEIGHTS["conciseness"] * conciseness
    )


def ragqa_cfc(gold, pred, trace=None):
    """CFC composite score (float in [0, 1]); bool >= threshold under tracing/bootstrap."""
    response = _response_text(pred)
    completeness, faithfulness, conciseness, _ = _cfc_components(
        gold.question, gold.response, response, _context_text(pred)
    )
    score = _cfc_total(completeness, faithfulness, conciseness)
    return score if trace is None else score >= CFC_THRESHOLD


def ragqa_cfc_feedback(output, example, trace=None, pred_name=None, pred_trace=None):
    """CFC composite score with textual feedback for GEPA / CodeEvolver reflection.

    Same CodeEvolver metric contract as the metrics above: the prediction is ``output``
    and the gold example row is ``example`` (fields ``question`` and ``response``). The
    retrieved passages are read off ``output.context`` (with a ``pred_trace`` fallback).
    """
    pred = output
    gold_question = _example_field(example, "question")
    gold_response = _example_field(example, "response")
    response = _response_text(pred)
    context_text = _context_text(pred, pred_trace)

    completeness, faithfulness, conciseness, meta = _cfc_components(
        gold_question, gold_response, response, context_text
    )
    score = _cfc_total(completeness, faithfulness, conciseness)

    parts = [
        f"CFC={score:.2f} "
        f"(completeness={completeness:.2f} [50%], "
        f"faithfulness={faithfulness:.2f} [25%], "
        f"conciseness={conciseness:.2f} [25%]).",
        f"Response is {meta['length_ratio']:.2f}x the length of the gold answer "
        f"(conciseness is exempt up to {CONCISENESS_EXEMPT_RATIO:g}x, full penalty at "
        f"{CONCISENESS_FULL_RATIO:g}x).",
        f"Question: {gold_question}",
        f"Gold response (reference): {gold_response}",
        f"System response: {response}",
    ]
    if meta["completeness_discussion"]:
        parts.append(f"Completeness analysis: {meta['completeness_discussion']}")
    if meta["groundedness_discussion"]:
        parts.append(f"Faithfulness analysis: {meta['groundedness_discussion']}")

    # Directional guidance: point the optimizer at the weakest component.
    weakest = min(
        ("completeness", completeness),
        ("faithfulness", faithfulness),
        ("conciseness", conciseness),
        key=lambda kv: kv[1],
    )[0]
    if weakest == "completeness":
        parts.append(
            "Completeness is the weakest component: the response omits key ideas present "
            "in the gold answer -- retrieve and cover more of the relevant content."
        )
    elif weakest == "faithfulness":
        if not meta["faith_grounded"]:
            parts.append(
                "Faithfulness is the weakest component AND no retrieved context was "
                "available -- ensure the program carries the retrieved passages on the "
                "prediction (pred.context) and grounds every claim in them."
            )
        else:
            parts.append(
                "Faithfulness is the weakest component: some claims are not supported by "
                "the retrieved passages -- ground every claim in the evidence and drop "
                "unsupported statements."
            )
    else:
        parts.append(
            "Conciseness is the weakest component: the response is too long relative to "
            "the gold answer -- say the same key ideas in fewer characters."
        )

    return ScoreWithFeedback(score=score, feedback=" ".join(parts))
