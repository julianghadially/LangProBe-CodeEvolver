"""Integration tests for pipeline modules.

Runs each pipeline against 5 training examples and asserts >20% metric score.
Requires OPENAI_API_KEY env var and network access to ColBERTv2 server.
"""

import json
from pathlib import Path

import dspy
import dspy.evaluate
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LM = dspy.LM("openai/gpt-4o-mini")


def load_train_examples(filename, input_keys, n=10):
    with open(DATA_DIR / filename) as f:
        raw = json.load(f)
    examples = []
    for item in raw[:n]:
        ex = dspy.Example(**item).with_inputs(*input_keys)
        examples.append(ex)
    return examples


# ── Hover ────────────────────────────────────────────────────────────────────


@pytest.fixture
def hover_examples():
    return load_train_examples("hoverBench_train.json", ["claim"])


def test_hover_multihop_predict_pipeline(hover_examples):
    from langProBe.hover.hover_pipeline import HoverMultiHopPredictPipeline
    from langProBe.hover.hover_utils import discrete_retrieval_eval

    pipeline = HoverMultiHopPredictPipeline()
    evaluator = dspy.Evaluate(
        devset=hover_examples,
        metric=discrete_retrieval_eval,
        num_threads=5,
        display_progress=True,
    )
    with dspy.context(lm=LM):
        result = evaluator(pipeline)

    score = result.score if hasattr(result, "score") else float(result)
    print(f"\nHoverMultiHopPredictPipeline score: {score}%")
    assert score >= 20.0, f"Expected >20% but got {score}%"


def test_hover_multihop_pipeline(hover_examples):
    from langProBe.hover.hover_pipeline import HoverMultiHopPipeline
    from langProBe.hover.hover_utils import discrete_retrieval_eval

    pipeline = HoverMultiHopPipeline()
    evaluator = dspy.Evaluate(
        devset=hover_examples,
        metric=discrete_retrieval_eval,
        num_threads=5,
        display_progress=True,
    )
    with dspy.context(lm=LM):
        result = evaluator(pipeline)

    score = result.score if hasattr(result, "score") else float(result)
    print(f"\nHoverMultiHopPipeline score: {score}%")
    assert score >= 20.0, f"Expected >=20% but got {score}%"


# ── HotpotQA ─────────────────────────────────────────────────────────────────


@pytest.fixture
def hotpot_examples():
    return load_train_examples("HotpotQABench_train.json", ["question"])


def test_hotpot_multihop_predict_pipeline(hotpot_examples):
    from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPredictPipeline

    pipeline = HotpotMultiHopPredictPipeline()
    evaluator = dspy.Evaluate(
        devset=hotpot_examples,
        metric=dspy.evaluate.answer_exact_match,
        num_threads=5,
        display_progress=True,
    )
    with dspy.context(lm=LM):
        result = evaluator(pipeline)

    score = result.score if hasattr(result, "score") else float(result)
    print(f"\nHotpotMultiHopPredictPipeline score: {score}%")
    assert score >= 20.0, f"Expected >=20% but got {score}%"


def test_hotpot_multihop_pipeline(hotpot_examples):
    from langProPlus.hotpotGEPA.hotpot_pipeline import HotpotMultiHopPipeline

    pipeline = HotpotMultiHopPipeline()
    evaluator = dspy.Evaluate(
        devset=hotpot_examples,
        metric=dspy.evaluate.answer_exact_match,
        num_threads=5,
        display_progress=True,
    )
    with dspy.context(lm=LM):
        result = evaluator(pipeline)

    score = result.score if hasattr(result, "score") else float(result)
    print(f"\nHotpotMultiHopPipeline score: {score}%")
    assert score >= 20.0, f"Expected >=20% but got {score}%"
