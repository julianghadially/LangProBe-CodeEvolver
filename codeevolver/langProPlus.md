# LangProPlus
Here, we added additional programs to the LangProBe benchmark.

Contents:
1. hotpotGEPA: This is a modified version of LangProBe Hot pot_QA, which reduces the number of retrieval steps.

## HotpotGEPA
Location: langProPlus/hotpotGEPA/hotpot_program.py

HotpotGEPA will mirrors the GEPA paper architecture, reconstructed to the best of our ability, based on this statement from GEPA:
> (Yang et al., 2018) is a large-scale question-answering dataset consisting of 113K Wikipedia-based
question-answer pairs. It features questions that require reasoning over multiple supporting documents. We modify the
last hop of the HoVerMultiHop program (described below) to answer the question instead of generating another query,
and the rest of the system remains unmodified. The textual feedback module identifies the set of relevant documents
remaining to be retrieved at each stage of the program, and provides that as feedback to the modules at that stage. We
use 150 examples for training, 300 for validation, and 300 for testing.

CodeEvolver will take this architecture as a starting point and optimize it. 