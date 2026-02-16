import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram, deduplicate

class ExtractQueryAspects(dspy.Signature):
    """Identify a distinct queryable aspect or entity from the claim that hasn't been covered yet.
    The aspect should be a specific entity, person, place, event, or concept that can be searched for.
    Ensure the aspect is different from the previous aspects already covered."""

    claim = dspy.InputField(desc="The claim to analyze")
    previous_aspects = dspy.InputField(desc="List of aspects already covered in previous hops")

    aspect = dspy.OutputField(desc="A distinct queryable aspect, entity, or concept from the claim")
    reasoning = dspy.OutputField(desc="Explanation of why this aspect is important and different from previous aspects")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15  # Retrieve 15 candidates per hop
        self.top_k = 7  # Keep top 7 after reranking
        self.extract_aspect = dspy.ChainOfThought(ExtractQueryAspects)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def rerank_passages(self, claim, passages):
        """Rerank passages using ColBERT scores with the claim as query.

        Args:
            claim: The claim text to use as query for scoring
            passages: List of passage strings to rerank

        Returns:
            Top 7 most relevant passages based on ColBERT scores
        """
        # Get the retrieval module from context
        rm = dspy.settings.rm
        if rm is None:
            # Fallback: return first top_k passages if no retriever available
            return passages[:self.top_k]

        # Use the retrieval module to score passages against the claim
        try:
            # Retrieve with a larger k to ensure we can find our passages in the results
            # We'll retrieve enough to hopefully include all our candidate passages
            results = rm(claim, k=100)

            if hasattr(results, 'passages'):
                retrieved_passages = results.passages
                # Build a mapping of passages to their rank (lower rank = better score)
                passage_to_rank = {p: i for i, p in enumerate(retrieved_passages)}

                # Score our candidate passages based on their rank in the retrieval results
                # Passages not found in results get a high penalty score
                scored_passages = []
                for p in passages:
                    rank = passage_to_rank.get(p, 999999)  # High penalty if not found
                    scored_passages.append((rank, p))

                # Sort by rank (lower is better) and take top_k
                scored_passages.sort(key=lambda x: x[0])
                reranked = [p for _, p in scored_passages[:self.top_k]]
                return reranked
            else:
                # Fallback if format is unexpected
                return passages[:self.top_k]
        except Exception as e:
            # Fallback on error: return first top_k passages
            return passages[:self.top_k]

    def forward(self, claim):
        # Track covered aspects for diversity
        covered_aspects = []
        all_docs = []

        # HOP 1: Extract and query the first aspect directly from the claim
        aspect1_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect1 = aspect1_result.aspect
        covered_aspects.append(aspect1)
        hop1_docs_raw = self.retrieve_k(aspect1).passages
        # Rerank using claim as query
        hop1_docs = self.rerank_passages(claim, hop1_docs_raw)
        all_docs.extend(hop1_docs)

        # HOP 2: Extract a second distinct aspect different from hop 1
        aspect2_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect2 = aspect2_result.aspect
        covered_aspects.append(aspect2)
        hop2_docs_raw = self.retrieve_k(aspect2).passages
        # Rerank using claim as query
        hop2_docs = self.rerank_passages(claim, hop2_docs_raw)
        all_docs.extend(hop2_docs)

        # HOP 3: Extract a third aspect or generate a connecting query
        aspect3_result = self.extract_aspect(
            claim=claim,
            previous_aspects=covered_aspects
        )
        aspect3 = aspect3_result.aspect
        hop3_docs_raw = self.retrieve_k(aspect3).passages
        # Rerank using claim as query
        hop3_docs = self.rerank_passages(claim, hop3_docs_raw)
        all_docs.extend(hop3_docs)

        # Deduplicate to ensure all documents are unique
        unique_docs = deduplicate(all_docs)

        return dspy.Prediction(retrieved_docs=unique_docs)
