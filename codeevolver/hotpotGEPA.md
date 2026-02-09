# hotpotGEPA Change Notes

## Source

GEPA paper (arXiv:2507.19457):
> "We modify the last hop of the HoVerMultiHop program to answer the question
> instead of generating another query, and the rest of the system remains unmodified."

> "The textual feedback module identifies the set of correct documents retrieved,
> and the set of documents remaining to be retrieved, and returns them as feedback text."

> "We use 150 examples for training, 300 for validation, and 300 for testing."

## Current State

`langProPlus/hotpotGEPA/` is an unmodified copy of `langProBe/hotpotQA/`. It contains
8 generic programs (Predict, CoT, RAG, SimplifiedBaleen, Archon variants) that have
nothing to do with the GEPA approach. These should be replaced.

## Program Change

The program is a copy of `HoverMultiHop` (`langProBe/hover/hover_program.py`) with
one modification: **replace Hop 3 (query generation + retrieval) with answer generation.**

### HoVerMultiHop (original, for reference)

```python
# langProBe/hover/hover_program.py — HoverMultiHop
def forward(self, claim):
    # HOP 1
    hop1_docs = self.retrieve_k(claim).passages
    summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

    # HOP 2
    hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
    hop2_docs = self.retrieve_k(hop2_query).passages
    summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

    # HOP 3
    hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
    hop3_docs = self.retrieve_k(hop3_query).passages

    return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
```

### HotpotMultiHop (GEPA adaptation)

Changes from HoVerMultiHop:
1. `claim` -> `question` in all signatures and the forward method
2. Remove `create_query_hop3` module entirely
3. Replace Hop 3 body with `generate_answer(question, summary_1, summary_2) -> answer`
4. Return `dspy.Prediction(answer=answer)` instead of `retrieved_docs`

```
HoVerMultiHop (HoVer):              HotpotMultiHop (HotpotQA):
  Input: claim                        Input: question
  Hop 1: Retrieve -> Summarize        Hop 1: Retrieve -> Summarize       [SAME]
  Hop 2: QueryGen -> Retrieve -> Sum  Hop 2: QueryGen -> Retrieve -> Sum [SAME]
  Hop 3: QueryGen -> Retrieve         Hop 3: GenerateAnswer              [CHANGED]
  Output: retrieved_docs              Output: answer
```

This gives exactly 2 retrieval calls per question, satisfying the constraint in
`requirements.md` ("Do NOT search more than two times per question").

## Textual Feedback Module

The textual feedback module is **not part of this program**. Per `requirements.md`:
> "CodeEvolver and the optimizer lives in its own separate repository."

The feedback module belongs in the CodeEvolver evaluator. It compares documents
retrieved at each hop against gold `supporting_facts` from the training data and
returns feedback text like: "Retrieved: [Doc A]. Still needed: [Doc B, Doc C]."

For this to work, `hotpot_data.py` should preserve `supporting_facts` (or extract
`gold_titles`) in the dspy.Example objects so the evaluator has access to them.

## Dataset

The existing `Benchmark` base class already enforces 150/300/300 splits for
train/dev/val, matching the GEPA paper. No changes needed to split logic.

`hotpot_data.py` change: preserve gold document titles from `supporting_facts`
for use by the evaluator's feedback module.

## Files to Change

| File | Change |
|------|--------|
| `hotpot_program.py` | Replace all 8 generic programs with `HotpotMultiHop` (and optionally a `Predict` variant). Copy `HoverMultiHop` from `langProBe/hover/hover_program.py`, rename `claim` -> `question`, replace hop 3 with answer generation. |
| `hotpot_data.py` | Preserve `gold_titles` from `supporting_facts` in dspy.Example objects. |
| `__init__.py` | Update program list and imports. |
| `hotpot_utils.py` | No changes needed in program repo. Feedback logic belongs in CodeEvolver. |
