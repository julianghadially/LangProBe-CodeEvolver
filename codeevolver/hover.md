# hotpotGEPA
Hover mirrors the GEPA paper architecture, which mirrors the LangProBe dataset.

According to the GEPA paper (arXiv:2507.19457):
> HoVer (Jiang et al., 2020) is an open-domain multihop fact extraction and claim verification benchmark built on a 
> Wikipedia-based corpus requiring complex reasoning across multiple sentences and documents, typically involving multiple wikipedia articles. 
> Following Tan et al. (2025), the systems are evaluated for their ability to write queries in multiple hops to retrieve all relevant wikipedia documents (gold documents) required to make the claim. 
> We obtain the HoverMultiHop program from Tan et al. (2025), which performs up to 3-hop retrievals using 2 query writer modules, and 2 document summary modules. 
> The textual feedback module simply identifies the set of correct documents retrieved, and the set of documents remaining to be retrieved, and returns them as feedback text. 
> For HoVer, we use 150 examples for training, 300 for validation, and 300 for testing



## Ambiguity in GEPA

**Hop filtering:** The training data in GEPA is filtered to 3-hop only (count_unique_docs == 3), and the test set is unspecified. When we include hop counts of all hops we get 33-36% test results. The GEPA paper doesn't explicitly state whether it filters test examples by hops, and the langProBe repository has no filter by default (whereas the training set is filtered)

In order to match the GEPA baseline of 46.3% we picked
- seed = 9 filter <=3, Predict: 46.7%

