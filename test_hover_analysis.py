import json
import dspy
from langProBe.hover.hover_program import HoverMultiHopPredict
from langProBe.hover.hover_utils import discrete_retrieval_eval

# Set up the retrieval model
COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"
rm = dspy.ColBERTv2(url=COLBERT_URL)

# Set up the language model (using a simple model for testing)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, rm=rm)

# Load the dev dataset
with open('data/hoverBench_dev.json', 'r') as f:
    data = json.load(f)

# Test on the first 3 examples as mentioned
program = HoverMultiHopPredict()

test_cases = [
    {
        "id": 1,
        "description": "Caerwys railway station claim",
        "needed": ["Afonwen", "A55 road", "Caerwys railway station"]
    },
    {
        "id": 2,
        "description": "20th Century Fox claim",
        "needed": ["Trax Colton", "20th Century Fox (2 entries)", "It Happened in Athens"]
    },
    {
        "id": 3,
        "description": "Warren Fu / Aaliyah claim",
        "needed": ["Warren Fu", "Aaliyah (with illegal marriage info)", "Soothe My Soul"]
    }
]

# Let's search for these specific examples in the dataset
print("Searching for examples matching the descriptions...\n")

for test_case in test_cases:
    print(f"=" * 80)
    print(f"Test Case {test_case['id']}: {test_case['description']}")
    print(f"Needed documents: {test_case['needed']}")
    print("=" * 80)

    # Find matching example
    found = False
    for idx, example in enumerate(data):
        # Check if any needed doc is in supporting facts
        supporting_keys = [sf['key'] for sf in example['supporting_facts']]

        # Match based on key terms in the claim
        if test_case['id'] == 1 and 'Caerwys' in example['claim']:
            found = True
        elif test_case['id'] == 2 and '20th Century Fox' in example['claim']:
            found = True
        elif test_case['id'] == 3 and 'Warren Fu' in example['claim']:
            found = True

        if found:
            print(f"\nFound at index {idx}")
            print(f"\nClaim: {example['claim']}")
            print(f"\nRequired supporting facts:")
            for sf in example['supporting_facts']:
                print(f"  - {sf['key']} (sentence {sf['value']})")

            # Run the program
            print(f"\n--- Running HoverMultiHopPredict ---")
            try:
                with dspy.context(rm=rm):
                    result = program(claim=example['claim'])

                print(f"\nTotal retrieved docs: {len(result.retrieved_docs)}")
                print(f"\nRetrieved document titles:")
                for i, doc in enumerate(result.retrieved_docs[:21], 1):
                    title = doc.split(" | ")[0]
                    print(f"  {i}. {title}")

                # Evaluate
                gold_titles = set([sf['key'] for sf in example['supporting_facts']])
                found_titles = set([doc.split(" | ")[0] for doc in result.retrieved_docs[:21]])

                print(f"\n--- Evaluation ---")
                print(f"Required titles: {gold_titles}")
                print(f"Found required titles: {gold_titles.intersection(found_titles)}")
                print(f"Missing required titles: {gold_titles - found_titles}")
                print(f"Score: {1.0 if gold_titles.issubset(found_titles) else 0.0}")

            except Exception as e:
                print(f"Error running program: {e}")
                import traceback
                traceback.print_exc()

            break

    if not found:
        print(f"\nExample not found in dataset!")

    print("\n")
