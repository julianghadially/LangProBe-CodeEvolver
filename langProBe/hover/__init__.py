from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop, HoverParallelEntityRetrieval
from .hover_pipeline import HoverMultiHopPipeline, HoverParallelEntityPipeline
from .hover_utils import discrete_retrieval_eval, MAX_RETRIEVED_DOCS

benchmark = [
    BenchmarkMeta(
        hoverBench, [HoverMultiHop()], discrete_retrieval_eval
    )
]

__all__ = [
    'HoverMultiHop',
    'HoverParallelEntityRetrieval',
    'HoverMultiHopPipeline',
    'HoverParallelEntityPipeline',
    'discrete_retrieval_eval',
    'MAX_RETRIEVED_DOCS',
    'hoverBench',
    'benchmark'
]
