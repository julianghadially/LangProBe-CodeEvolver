import dspy
import dspy.evaluate
from langProBe.benchmark import BenchmarkMeta
import langProBe.dspy_program as dspy_program

from .archive.RAGQAArenaTech_data import RAGQAArenaBench
from .RAGQAArenaTech_program import SimplifiedBaleen
from .RAGQAArenaTech_retrieval import get_default_retriever

eval_lm = dspy.LM("openai/gpt-4o")
eval_module = dspy.evaluate.SemanticF1()
eval_module.set_lm(eval_lm)

# Program instances for the LangProBe benchmark registry. SimplifiedBaleen gets
# the shared retriever (one index load); the others are retrieval-free baselines.
basic_signature = "question -> response"

RAGQAPredict = dspy_program.Predict(basic_signature)
RAGQACoT = dspy_program.CoT(basic_signature)
RAGQASimplifiedBaleen = SimplifiedBaleen(get_default_retriever())
RAGQAGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(basic_signature)
RAGQAGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(basic_signature, n=20)
RAGQAGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(basic_signature)
RAGQAGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    basic_signature, n=20
)

benchmark = [
    BenchmarkMeta(
        RAGQAArenaBench,
        [
            RAGQASimplifiedBaleen,
            RAGQACoT,
            RAGQAPredict,
            RAGQAGeneratorCriticRanker,
            RAGQAGeneratorCriticFuser,
            RAGQAGeneratorCriticFuser_20,
            RAGQAGeneratorCriticRanker_20,
        ],
        eval_module,
    )
]
