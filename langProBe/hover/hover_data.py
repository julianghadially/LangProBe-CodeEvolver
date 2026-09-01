from ..benchmark import Benchmark
import dspy
import random
from .hover_utils import count_unique_docs


class hoverBench(Benchmark):
    def init_dataset(self):
        # Lazy: `datasets` (HF hub builder) and `tqdm` live in
        # requirements-full.txt, not requirements.txt. This package's __init__
        # imports this module, and the CodeEvolver path (hover_pipeline) only
        # needs the committed data/hoverBench_*.json files -- never the builder.
        from datasets import load_dataset
        import tqdm

        dataset = load_dataset("hover-nlp/hover", revision="refs/convert/parquet")

        hf_trainset = dataset["train"]
        hf_testset = dataset[
            "validation"
        ]  # Using validation dataset because test dataset is not labeled

        reformatted_hf_trainset = []
        reformatted_hf_testset = []

        for example in tqdm.tqdm(hf_trainset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) == 3:  # Limit to 3 hop examples
                reformatted_hf_trainset.append(
                    dict(claim=claim, supporting_facts=supporting_facts, label=label)
                )

        for example in tqdm.tqdm(hf_testset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) <= 3:  # Limit to 3 hop examples
                reformatted_hf_testset.append(
                    dict(claim=claim, supporting_facts=supporting_facts, label=label)
                )

        rng = random.Random()
        rng.seed(0)
        rng.shuffle(reformatted_hf_trainset)
        rng = random.Random()
        rng.seed(9) 
        rng.shuffle(reformatted_hf_testset)

        trainset = reformatted_hf_trainset
        testset = reformatted_hf_testset

        trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]
        testset = [dspy.Example(**x).with_inputs("claim") for x in testset]

        self.dataset = trainset
        self.test_set = testset
