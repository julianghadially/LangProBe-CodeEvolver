# hotpotGEPA
HotpotGEPA mirrors the GEPA paper architecture, reconstructed to the best of our ability.

According to the GEPA paper (arXiv:2507.19457):
> "We modify the last hop of the HoVerMultiHop program to answer the question
> instead of generating another query, and the rest of the system remains unmodified."
> "The textual feedback module identifies the set of correct documents retrieved, and the set of documents remaining to be retrieved, and returns them as feedback text." 
> "We use 150 examples for training, 300 for validation, and 300 for testing."

## Location
`langProPlus/hotpotGEPA/`

## Architecture

Adapted from `HoverMultiHop` (`langProBe/hover/hover_program.py`). The only structural change is replacing Hop 3 (query + retrieval) with answer generation.

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

Two program variants are included: `HotpotMultiHop` (ChainOfThought) and `HotpotMultiHopPredict` (Predict).

## Dataset

Uses HotpotQA fullwiki from HuggingFace. The `Benchmark` base class enforces 150/300/300 splits for train/dev/val, matching the GEPA paper. `hotpot_data.py` preserves `gold_titles` from `supporting_facts` for use by the evaluator's feedback module.

## Textual Feedback Module

The textual feedback module is not part of this program. It belongs in the CodeEvolver / GEPA repository.

## Ambiguity in GEPA

We tried to get close to the GEPA baseline of 38% on gpt 4.1 mini.  
- seed = 6, Predict: 34.3%
