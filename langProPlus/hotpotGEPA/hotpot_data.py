from langProBe.benchmark import Benchmark
import dspy
from datasets import load_dataset


class HotpotQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("hotpot_qa", "fullwiki")
        self.dataset = [
            dspy.Example(
                question=x["question"],
                answer=x["answer"],
                gold_titles=list(set(x["supporting_facts"]["title"])),
            ).with_inputs("question")
            for x in raw_datasets["train"]
        ]
        self.test_set = [
            dspy.Example(
                question=x["question"],
                answer=x["answer"],
                gold_titles=list(set(x["supporting_facts"]["title"])),
            ).with_inputs("question")
            for x in raw_datasets["validation"]
        ]
