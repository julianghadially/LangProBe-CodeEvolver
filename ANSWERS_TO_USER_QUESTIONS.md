# Answers to User Questions: Evaluation Feedback Analysis

## Background
You asked about analyzing 4 specific examples with score 0.0. Based on my analysis, I found these examples are in the **training set** (not test set), and I've analyzed both those specific examples and broader failure patterns across the dataset.

---

## Question 1: Example 1 - Burn Gorman, "Crimson Peak", and Guillermo del Toro

**Claim**: "Australian Burn Gorman played Mr. Holly in a 2015 American gothic romance film. The film was co-produced by the director Guillermo del Toro who also directed 'Pacific Rim'."

**Supporting facts needed**:
- Guy Davis (comics)
- Crimson Peak
- Burn Gorman

**Are the right documents being retrieved?**
- ✓ **Crimson Peak**: YES - Retrieved multiple times (positions 1, 9, 15)
- ✓ **Burn Gorman**: YES - Retrieved multiple times (positions 8, 16)
- ✗ **Guy Davis (comics)**: NO - Not found in any of the 21 retrieved documents

**What's going wrong?**
- The retrieval successfully finds the movie and the actor
- But it completely misses "Guy Davis (comics)" - a comic book artist
- The claim doesn't explicitly mention Guy Davis, so the connection is unclear
- All retrieved documents focus on the film and actors, never branching to comic book artists

**Pattern**: This is a **missing conceptual connection** - the claim has a hidden requirement (Guy Davis) that isn't directly mentioned, and the retrieval never discovers this entity.

**Status**: When I ran this example (index 56 in training set), it **FAILED** with score 0.0 (2/3 documents found).

---

## Question 2: Example 2 - Godhead band and Joe Lynn Turner

**Claim**: "The person that signed Godhead (band) to their label and Joe Lynn Turner are both singer, songwriters."

**Supporting facts needed**:
- Marilyn Manson
- Joe Lynn Turner
- Godhead (band)

**Are the right documents being retrieved?**
- ✓ **Godhead (band)**: YES - Retrieved multiple times (positions 1, 15)
- ✓ **Marilyn Manson**: YES - Retrieved multiple times (positions 8, 16)
- ✓ **Joe Lynn Turner**: YES - Retrieved (position 20)

**What's going wrong?**
- **NOTHING!** All required documents are present in the top 21.

**Pattern**: This example actually **SUCCEEDS**.

**Status**: When I ran this example (index 148 in training set), it **PASSED** with score 1.0 (3/3 documents found).

---

## Question 3: Example 3 - François de Fleury and Battle of Monmouth

**Claim**: "Francois de Fleury served in a cavalry battle in the American Revolutionary War in June of 1778. Henry Monckton was killed during that battle."

**Supporting facts needed**:
- François de Fleury
- Battle of Monmouth
- Henry Monckton

**Are the right documents being retrieved?**
- ✓ **François de Fleury**: YES - Retrieved multiple times (positions 1, 16)
- ✓ **Battle of Monmouth**: YES - Retrieved multiple times (positions 10, 18)
- ✓ **Henry Monckton**: YES - Retrieved multiple times (positions 3, 8, 15)

**What's going wrong?**
- **NOTHING!** All required documents are present in the top 21.

**Pattern**: This example actually **SUCCEEDS**.

**Status**: When I ran this example (index 55 in training set), it **PASSED** with score 1.0 (3/3 documents found).

---

## Question 4: Example 4 - G.I. Joe Hall of Fame and Channing Tatum

**Claim**: "The G.I. Joe: Hall of Fame includes a nostalgic character demanded by fans. The fictional character is portrayed by an actor who made his film debut in the 2005 film Coach Angereds Teater."

**Supporting facts needed**:
- Duke (G.I. Joe)
- Channing Tatum
- G.I. Joe: Hall of Fame

**Are the right documents being retrieved?**
- ✓ **G.I. Joe: Hall of Fame**: YES - Retrieved (position 1)
- ✓ **Duke (G.I. Joe)**: YES - Retrieved multiple times (positions 3, 19)
- ✓ **Channing Tatum**: YES - Retrieved multiple times (positions 11, 16)

**What's going wrong?**
- **NOTHING!** All required documents are present in the top 21.

**Pattern**: This example actually **SUCCEEDS**.

**Status**: When I ran this example (index 23 in training set), it **PASSED** with score 1.0 (3/3 documents found).

---

## Summary: Are the Right Documents Being Retrieved?

### Based on your 4 examples:
1. **Burn Gorman**: ❌ NO - Missing "Guy Davis (comics)"
2. **Godhead**: ✅ YES - All 3 documents found
3. **François de Fleury**: ✅ YES - All 3 documents found
4. **G.I. Joe**: ✅ YES - All 3 documents found

**Only 1 out of 4 of your specific examples failed when I tested them.**

---

## Why the Discrepancy?

You mentioned "all 4 examples have score 0.0", but when I ran them:
- 3 succeeded (Godhead, François de Fleury, G.I. Joe)
- 1 failed (Burn Gorman)

**Possible reasons**:
1. **Different model/configuration**: You may have evaluated with a different LM that produces different queries
2. **Non-deterministic queries**: The ChainOfThought modules may generate different queries on different runs
3. **Different dataset split**: The examples might be from a different split or version
4. **Retrieval API changes**: The ColBERTv2 API results may have changed

---

## Broader Pattern Analysis (Test Set)

I also analyzed the **test set** (first 20 examples) and found clear failure patterns:

### Overall Statistics:
- **Success rate**: 65% (13/20 examples)
- **Failure rate**: 35% (7/20 examples)

### Main Failure Patterns:

#### 1. **Query Drift / Topic Lock-in** (Most Common)
**Example**: Jack Kevorkian claim
- Required: Jack Kevorkian, Christy Canyon, Playboy Radio
- Retrieved: 21 documents ALL about Jack Kevorkian and related names
- Problem: System got "stuck" on one entity, never explored the "porn star" aspect

#### 2. **Missing Conceptual Knowledge**
**Example**: Greek alphabet claim
- Required: Antonis Fotsis, Ilysiakos B.C., Greek alphabet
- Retrieved: 21 documents about Greek soccer players/clubs
- Problem: Never understood "alphabet derived from Phoenician" = Greek alphabet

#### 3. **Entity Disambiguation Failure**
**Example**: Jiang Wen claim
- Required: Marcel Duchamp, Prelude to a Broken Arm, Jiang Wen
- Retrieved: Jiang Wei, Jiang Zhaohe, Jiang Gan (wrong people!)
- Problem: Retrieved similar names but not the correct entity

---

## Are Documents There But Buried?

**NO** - in failure cases, the required documents are **completely absent** from the top 21 results.

This is not a ranking problem where the right document is at position 30. The documents are nowhere in the retrieved set at all.

**Evidence**:
- Example 0 (Greek alphabet): 0 occurrences of "Greek alphabet" in 21 docs
- Example 2 (Christy Canyon): 0 occurrences of "Christy Canyon" or "Playboy Radio" in 21 docs
- Example 1 (Jiang Wen): Retrieved 7 different "Jiang" people, but not Jiang Wen

---

## Root Cause: The Pattern

The pattern is **query generation failure**. The multi-hop system:

1. **Hop 1**: Successfully retrieves documents for the most prominent entity in the claim
2. **Hop 2**: Generates a query based on Hop 1 summary, but often just retrieves MORE documents about the same entity
3. **Hop 3**: Generates a query based on Hops 1+2, continues the same narrow focus

**The queries are not diversifying enough** to cover all aspects of multi-topic claims.

### Why This Happens:

1. **Summarization narrows focus**: Summaries emphasize the dominant entity, losing information about secondary topics
2. **No coverage tracking**: System doesn't know which claim aspects are unaddressed
3. **No explicit diversification**: Queries don't explicitly try to explore different aspects
4. **Retrieval model limitations**: Similar entity names confuse the retrieval (wrong "Jiang")

---

## Key Insight

The system is often "close but not close enough":
- **67% coverage** (2/3 docs) → **Score: 0.0** (failure)
- **33% coverage** (1/3 docs) → **Score: 0.0** (failure)

The all-or-nothing metric means even small gaps result in complete failure.

---

## Recommendation

The core issue is **insufficient query diversification in the multi-hop chain**. The system needs to:

1. Track which entities/concepts have been covered
2. Generate queries that explicitly target uncovered aspects
3. Detect when stuck in one topic cluster
4. Add diversity/exploration mechanisms to break out of query drift

This is primarily a **prompt engineering / query generation problem**, not a retrieval model problem.
