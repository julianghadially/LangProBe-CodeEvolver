from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop, HoverMultiHopPredict
from .hover_utils import discrete_retrieval_eval, label_accuracy_eval

benchmark = [
    BenchmarkMeta(hoverBench, [HoverMultiHop()], discrete_retrieval_eval),
    BenchmarkMeta(hoverBench, [HoverMultiHopPredict()], label_accuracy_eval),
]
