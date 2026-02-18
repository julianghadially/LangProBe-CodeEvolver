from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop
from .hover_utils import discrete_retrieval_eval

benchmark = [
    BenchmarkMeta(
        hoverBench,
        [
            HoverMultiHop(use_reranking=False, retrieve_k=7, final_k=7),  # Baseline
            HoverMultiHop(use_reranking=True, retrieve_k=30, final_k=7),  # Reranking
        ],
        discrete_retrieval_eval
    )
]
