# Visual Examples of Retrieval Failures

## Example 0: The "Greek Alphabet" Problem
### Conceptual Knowledge Gap

**Claim**: "Antonis Fotsis is a player for the club who's name has the starting letter from an alphabet derived from the Phoenician alphabet."

**Required Documents**:
- ✓ Antonis Fotsis
- ✓ Ilysiakos B.C.
- ✗ **Greek alphabet** ← MISSING

**What Was Retrieved (21 documents)**:
```
Hop 1:
 1. ✓ Antonis Fotsis
 2. Antonis Fotiadis
 3. Fotis Konstantinidis
 4. Fotios Papoulis
 5. Antonis Antoniadis
 6. Fotios Vasilopoulos
 7. Cippi of Melqart

Hop 2:
 8. Antonis Fotiadis
 9. Antonis Fotsis
10. Antonis Georgallides
11. Antonis Dedakis
12. Antonis Rikka
13. Antonis Bourselis
14. Antonios Vouzas

Hop 3:
15. Antonis Fotsis
16. Antonis Fotiadis
17. Fokikos A.C.
18. Platanias F.C.
19. ✓ Ilysiakos B.C.
20. Loizos Kakoyiannis
21. Antonis Makris
```

**Problem**: The system retrieved many Greek soccer-related documents but never understood that "alphabet derived from the Phoenician alphabet" refers to the Greek alphabet. All queries focused on the person and club, missing the conceptual/linguistic aspect entirely.

---

## Example 1: The "Wrong Jiang" Problem
### Entity Disambiguation Failure

**Claim**: "I would be more worried about playing a chess game against the north Belgian artist that made Prelude to a Broken Arm, than Jiang Wen."

**Required Documents**:
- ✓ Prelude to a Broken Arm
- ✓ Marcel Duchamp
- ✗ **Jiang Wen** ← MISSING

**What Was Retrieved (21 documents)**:
```
Hop 1:
 1. ✓ Prelude to a Broken Arm
 2. Zeng Fanzhi
 3. Jiang Zhaohe          ← Wrong Jiang!
 4. Jiang Jin              ← Wrong Jiang!
 5. Wen Yang (Three Kingdoms)
 6. Jiang Fengzhi          ← Wrong Jiang!
 7. Liu Wenzhe

Hop 2:
 8. ✓ Marcel Duchamp
 9. Fei Yi
10. Jiang Wei              ← Wrong Jiang!
11. Pierre Pinoncelli
12. Qiao Shi
13. Jiang Tianyong         ← Wrong Jiang!
14. Li Tianyou

Hop 3:
15. Étant donnés
16. Arimaa
17. Jiang Wei's Northern Expeditions  ← Wrong Jiang!
18. Fei Yi
19. Lü Jiangang
20. Jiang Gan              ← Wrong Jiang!
21. Ye Jiangchuan
```

**Problem**: The system found many people with "Jiang" in their name but not the correct one (Jiang Wen). This shows the retrieval is getting semantically close but failing at precise entity matching.

---

## Example 2: The "Query Drift" Problem
### Stuck on One Entity

**Claim**: "Jacob 'Jack' Kevorkian, born in 1977, is best known for publicly championing a terminal patient's right to die via physician-assisted suicide. Not the Hall of Fame porn star that replaced Juli Ashton on Playboy Radio."

**Required Documents**:
- ✓ Jack Kevorkian
- ✗ **Christy Canyon** ← MISSING
- ✗ **Playboy Radio** ← MISSING

**What Was Retrieved (21 documents)**:
```
Hop 1:
 1. ✓ Jack Kevorkian
 2. Jacob Bolotin
 3. Kenneth C. Edelin
 4. H. Jack Geiger
 5. Jacob Sheskin
 6. Jacob Liboschütz
 7. Jack Kershaw

Hop 2:
 8. Jack Kevorkian (duplicate)
 9. Hagop Kevorkian         ← Related name
10. N. Gregory Hamilton
11. Ivan Gevorkian          ← Related name
12. Maurice Généreux
13. Vahram Kevorkian        ← Related name
14. Jack W. Szostak

Hop 3:
15. Jack Kevorkian (duplicate)
16. Vahram Kevorkian (duplicate)
17. Michigan gubernatorial election, 1998
18. Albert Kapikian
19. Ivan Gevorkian (duplicate)
20. Paul Batista
21. You Don't Know Jack (film)
```

**Problem**: The retrieval system got "stuck" on Jack Kevorkian. All 21 documents are about medical professionals, people with similar names, or Kevorkian-related topics. The system never pivoted to "Hall of Fame porn star", "Christy Canyon", or "Playboy Radio" - the contrasting part of the claim.

**Query Drift Pattern**:
- Hop 1: Found Kevorkian → Good!
- Hop 2: Query generated from "Kevorkian summary" → Retrieves more Kevorkians
- Hop 3: Query generated from "more Kevorkians summary" → Still more Kevorkians
- Result: 21 documents, all Kevorkian-related, missing the entire "porn star" angle

---

## Pattern Summary

### 1. Conceptual Knowledge Gap (Example 0)
- **Pattern**: Missing abstract concepts or background knowledge
- **All retrieved docs**: Stay within one domain (Greek soccer)
- **Missing**: Conceptual connection (Greek alphabet)

### 2. Entity Disambiguation Failure (Example 1)
- **Pattern**: Retrieving similar but wrong entities
- **Retrieved**: Jiang Wei, Jiang Zhaohe, Jiang Gan, etc.
- **Missing**: The correct entity (Jiang Wen)

### 3. Query Drift / Topic Lock-in (Example 2)
- **Pattern**: Multi-topic claims where retrieval fixates on one topic
- **Retrieved**: 21 documents all about one entity
- **Missing**: Entire secondary topic (Christy Canyon, Playboy Radio)

---

## Why This Matters

The evaluation metric requires **ALL** supporting documents to be found. Even retrieving 2 out of 3 counts as complete failure (score = 0.0).

This means:
- Example 0: 67% coverage (2/3) → **Score: 0.0**
- Example 1: 67% coverage (2/3) → **Score: 0.0**
- Example 2: 33% coverage (1/3) → **Score: 0.0**

The system is often "close" but not achieving 100% coverage needed for success.

---

## Success Example for Contrast

### Example 3: Success Case

**Claim**: "Lili Chookasian was born 7 days before the conductor for Where Are You My Brothers?."

**Required Documents**:
- ✓ Lili Chookasian
- ✓ Constantine Orbelian
- ✓ Where Are You My Brothers?

**What Was Retrieved**:
```
Hop 1:
 1. ✓ Lili Chookasian
 2. ✓ Where Are You My Brothers?
 3. Anamika Choudhari
 4. Ruben Liljefors
 5. 702 (group)
 6. Ingemar Liljefors
 7. Choo Hoey

Hop 2:
 8. ✓ Constantine Orbelian
 9. Lili Chookasian (duplicate)
10. Konstantin Orbelyan
11. György Orbán
12. Desiderius Orban
13. Stephen Orbelian
14. Leon Orbeli

Hop 3:
15. Constantine Orbelian (duplicate)
16. Anna Orbeliani, Queen Consort of Imereti
17. Lili Chookasian (duplicate)
18. Lilian Constantini
19. Constantine Ypsilantis
20. Theodora Komnene (daughter of Alexios I)
21. Doina Șnep-Bălan
```

**Why It Succeeded**:
- All entities are closely related (Armenian/classical music figures)
- Natural semantic connections between the documents
- Retrieval stayed in coherent topic space
- Found all 3 required documents within 21 results

**Key Difference**: In success cases, the required documents are semantically close and naturally co-occur. In failure cases, the claim connects disparate topics (soccer player → Greek alphabet, or doctor → porn star) that don't naturally retrieve together.
