from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop, HoverMultiHopWithVerification
from .hover_verifier import StructuredClaimVerifier
from .hover_utils import discrete_retrieval_eval, verification_eval

benchmark = [
    # Original retrieval-only benchmark
    BenchmarkMeta(hoverBench, [HoverMultiHop()], discrete_retrieval_eval),
    # New verification benchmark
    BenchmarkMeta(
        hoverBench,
        [HoverMultiHopWithVerification()],
        verification_eval,
        name="hoverVerification",
    ),
]
