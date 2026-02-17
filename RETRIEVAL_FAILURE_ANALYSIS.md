# Retrieval Failure Pattern Analysis

## Overview
Analysis of the HoVer multi-hop fact verification system's retrieval failures based on test set evaluation.

## Evaluation Metrics
- **Success Rate**: 65% (13/20 examples in test sample)
- **Failure Rate**: 35% (7/20 examples)
- **Metric**: All supporting documents must be retrieved within top 21 results (7 docs × 3 hops)

## Key Failure Patterns

### Pattern 1: Query Drift in Multi-Hop Chain
**Frequency**: Most common failure mode

**Symptom**: The retrieval chain focuses heavily on one entity and fails to "hop" to related entities.

**Example 1 - Jack Kevorkian**
- **Claim**: About Jack Kevorkian AND Christy Canyon (porn star) AND Playboy Radio
- **Retrieved**: All 21 documents heavily focused on Jack Kevorkian and related names
- **Missing**: `Christy Canyon`, `Playboy Radio`
- **Problem**: After retrieving Jack Kevorkian in Hop 1, the system generated queries that continued focusing on Kevorkian-related entities (Hagop Kevorkian, Vahram Kevorkian, etc.) rather than pivoting to the "porn star" and "Playboy Radio" aspects of the claim

**Example 2 - Jiang Wen**
- **Claim**: About Marcel Duchamp (Belgian artist) AND Prelude to a Broken Arm AND Jiang Wen
- **Retrieved**: Found Marcel Duchamp and Prelude to a Broken Arm
- **Missing**: `Jiang Wen`
- **Problem**: Retrieved many similar Chinese names (Jiang Wei, Jiang Zhaohe, Jiang Gan, etc.) but not the correct one. The queries appear to be searching for "Jiang" + generic terms rather than the specific person.

### Pattern 2: Missing Conceptual/Background Knowledge
**Frequency**: Common in abstract or conceptual requirements

**Example - Greek Alphabet**
- **Claim**: About a club name starting with a letter from "an alphabet derived from the Phoenician alphabet"
- **Retrieved**: Found `Antonis Fotsis` and `Ilysiakos B.C.`
- **Missing**: `Greek alphabet`
- **Problem**: The claim refers to the Greek alphabet conceptually (as derived from Phoenician), but the retrieval system never generated a query targeting "Greek alphabet" directly. All 21 documents focused on Greek soccer players and clubs, missing the linguistic/historical connection entirely.

### Pattern 3: Multi-Topic Claims with Distant Connections
**Frequency**: Occurs when claims connect disparate topics

**Example - Playboy Radio + Jack Kevorkian**
- The claim creates a contrast between a medical ethics figure and an adult entertainment personality
- The retrieval hops didn't successfully bridge from "physician-assisted suicide" to "porn star" to "Playboy Radio"
- The query generation seems to lack the semantic understanding that "Hall of Fame porn star" should lead to retrieving Christy Canyon's page

## Root Cause Analysis

### 1. **Query Generation Lacks Topic Diversity**
The `create_query_hop2` and `create_query_hop3` modules use ChainOfThought to generate new queries based on:
- Original claim
- Previous summaries

However, the queries appear to be:
- Too focused on entities already retrieved
- Not explicitly targeting missing aspects of the claim
- Generating related but incorrect entities (e.g., similar Chinese names instead of the specific person)

### 2. **Summarization Narrows Focus**
The summarization step may be:
- Emphasizing dominant entities (e.g., "Jack Kevorkian")
- Losing information about secondary entities needed (e.g., "Christy Canyon", "Playboy Radio")
- Not preserving the multi-topic nature of claims

### 3. **No Explicit Coverage Tracking**
The system doesn't track which parts of the claim have been covered:
- No mechanism to identify that "Greek alphabet" aspect is unaddressed
- No detection that all retrieved docs are about one entity
- No diversification strategy for queries

### 4. **Retrieval Model Limitations**
ColBERTv2 retrieval may be:
- Returning semantically similar but factually incorrect results (wrong "Jiang" person)
- Not distinguishing between name variants well enough
- Overfitting to the most prominent entity in the query

## Distribution of Missing Documents
- **1 missing document**: 5 examples (71%)
- **2 missing documents**: 2 examples (29%)

Most failures are close to success, missing just 1-2 supporting documents out of 3-5 needed.

## Specific Failure Examples

### Example 0: Antonis Fotsis
- **Missing**: Greek alphabet
- **Pattern**: Conceptual knowledge gap
- **All 21 docs**: Greek soccer players and clubs, no linguistics/history

### Example 1: Marcel Duchamp
- **Missing**: Jiang Wen
- **Pattern**: Wrong entity (retrieved similar Chinese names)
- **Retrieved similar**: Jiang Wei, Jiang Zhaohe, Jiang Gan

### Example 2: Jack Kevorkian
- **Missing**: Christy Canyon, Playboy Radio
- **Pattern**: Query drift - stuck on medical ethics figure
- **All 21 docs**: Medical professionals and Kevorkian family members

### Example 4: Zohra Lampert
- **Missing**: Elia Kazan
- **Pattern**: Multi-hop failure - didn't reach the director

### Example 6: Rudolf Eucken
- **Missing**: Howard Fast, Hessian
- **Pattern**: Multi-topic failure (Nobel laureate + different author)

### Example 8: Elvis Presley
- **Missing**: Elvis Presley himself
- **Pattern**: Retrieved album but not artist (surprising!)

### Example 19: Lovelace film
- **Missing**: One Percent More Humid
- **Pattern**: Failed to retrieve all films featuring the actress

## Recommendations

### 1. **Improve Query Diversification**
- Add explicit instructions to generate queries targeting different aspects of the claim
- Track which entities/concepts have been retrieved
- Generate queries that explicitly avoid already-retrieved topics

### 2. **Enhance Summarization**
- Preserve multi-topic structure in summaries
- Explicitly note which aspects of the claim remain unaddressed
- Include "missing information" in the summary passed to next hop

### 3. **Add Coverage Feedback**
- After each hop, check which claim entities are still missing
- Use this to guide next query generation
- Consider a "gap analysis" step between hops

### 4. **Query Formulation Improvements**
- For ambiguous entity references (e.g., "north Belgian artist"), include more context from claim
- For conceptual connections (e.g., "alphabet derived from Phoenician"), make these explicit in queries
- Add negative examples to avoid wrong entities (e.g., "Jiang Wen not Jiang Wei")

### 5. **Consider Query Reformulation**
- If Hop 2 retrieves documents very similar to Hop 1, trigger a reformulation
- Detect when stuck in one topic cluster
- Add exploration vs exploitation strategy

## Success Cases (for contrast)

Example 3 (Success):
- **Claim**: About Lili Chookasian, Constantine Orbelian, "Where Are You My Brothers?"
- **Result**: Found all 3 documents
- **Why it worked**: All entities are closely related (Armenian musicians), retrieval stayed on topic

Example 7 (Success):
- **Claim**: About Qutebrowser and DuckDuckGo
- **Result**: Found both documents
- **Why it worked**: Direct technical relationship, only 2 docs needed
