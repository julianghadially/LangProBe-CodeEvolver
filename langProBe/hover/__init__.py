from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop
from .hover_utils import discrete_retrieval_eval

benchmark = [
    BenchmarkMeta(
        hoverBench, [HoverMultiHop()], discrete_retrieval_eval
    )
]
