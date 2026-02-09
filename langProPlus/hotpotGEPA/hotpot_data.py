from langProBe.benchmark import Benchmark
import dspy
import random
from datasets import load_dataset


class HotpotQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("hotpot_qa", "fullwiki")
        trainset = [
            dspy.Example(
                question=x["question"],
                answer=x["answer"],
                gold_titles=list(set(x["supporting_facts"]["title"])),
            ).with_inputs("question")
            for x in raw_datasets["train"]
        ]
        testset = [
            dspy.Example(
                question=x["question"],
                answer=x["answer"],
                gold_titles=list(set(x["supporting_facts"]["title"])),
            ).with_inputs("question")
            for x in raw_datasets["validation"]
        ]

        rng = random.Random()
        rng.seed(0)
        rng.shuffle(trainset)
        rng = random.Random()
        rng.seed(6)
        rng.shuffle(testset)

        self.dataset = trainset
        self.test_set = testset
