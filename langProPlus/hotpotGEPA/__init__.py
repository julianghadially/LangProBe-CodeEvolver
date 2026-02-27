import dspy.evaluate
from .hotpot_data import HotpotQABench
from .hotpot_program import HotpotMultiHop
from .hotpot_pipeline import HotpotMultiHopPipeline
from langProBe.benchmark import BenchmarkMeta

exact_match_metric = dspy.evaluate.answer_exact_match

benchmark = [
    BenchmarkMeta(
        HotpotQABench,
        [HotpotMultiHop(), HotpotMultiHopPipeline()],
        dspy.evaluate.answer_exact_match,
    )
]
