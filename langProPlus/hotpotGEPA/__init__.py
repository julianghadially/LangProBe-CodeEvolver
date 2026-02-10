import dspy.evaluate
from .hotpot_data import HotpotQABench
from .hotpot_program import HotpotMultiHop, HotpotMultiHopPredict
from langProBe.benchmark import BenchmarkMeta

exact_match_metric = dspy.evaluate.answer_exact_match

benchmark = [
    BenchmarkMeta(
        HotpotQABench,
        [HotpotMultiHop(), HotpotMultiHopPredict()],
        dspy.evaluate.answer_exact_match,
    )
]
