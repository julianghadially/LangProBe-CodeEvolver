from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop, HoverEntityAwareMultiHop
from .hover_utils import discrete_retrieval_eval

benchmark = [
    BenchmarkMeta(
        hoverBench, [HoverMultiHop()], discrete_retrieval_eval
    )
]

# Alternative benchmark with entity-aware retrieval
entity_aware_benchmark = [
    BenchmarkMeta(
        hoverBench, [HoverEntityAwareMultiHop()], discrete_retrieval_eval
    )
]
