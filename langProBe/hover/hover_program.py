import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ClaimAspectIdentifier(dspy.Signature):
    """Decompose the claim into 2-4 distinct aspects, entities, or concepts that need evidence.
    Each aspect should represent a key component that requires separate verification or supporting documents.
    Focus on identifying the main entities, events, relationships, or properties mentioned in the claim."""

    claim = dspy.InputField(desc="The claim to decompose into aspects")
    aspects: list[str] = dspy.OutputField(
        desc="List of 2-4 distinct aspects/entities/concepts from the claim that need evidence"
    )


class AspectTargetedQueryGenerator(dspy.Signature):
    """Generate a search query that specifically targets an aspect of the claim that hasn't been covered yet.
    The query should be designed to retrieve documents relevant to the target aspect, taking into account
    which aspects have already been covered by previous searches."""

    claim = dspy.InputField(desc="The original claim being verified")
    target_aspect = dspy.InputField(desc="The specific aspect to target with this query")
    covered_aspects = dspy.InputField(desc="Aspects that have already been covered by previous searches")
    query = dspy.OutputField(desc="Search query targeting the uncovered aspect")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using aspect-aware coverage tracking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    APPROACH
    - Decomposes the claim into 2-4 key aspects that need evidence
    - Tracks which aspects have been covered after each retrieval hop
    - Generates targeted queries for uncovered aspects rather than summarizing previous results
    - Ensures query diversification through explicit aspect coverage tracking'''

    def __init__(self):
        super().__init__()
        self.k = 7  # Retrieve 7 documents per hop for 21 total
        self.aspect_identifier = dspy.ChainOfThought(ClaimAspectIdentifier)
        self.query_generator = dspy.ChainOfThought(AspectTargetedQueryGenerator)
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def _check_aspect_coverage(self, aspect, retrieved_docs):
        """Check if an aspect is covered by analyzing retrieved document titles/content.
        Returns True if the aspect appears to be covered by the documents."""
        aspect_lower = aspect.lower()
        aspect_terms = set(aspect_lower.split())

        for doc in retrieved_docs:
            # Extract title from document (format: "title | content")
            doc_text = doc.lower()
            doc_terms = set(doc_text.split())

            # Check if significant overlap exists (simple heuristic)
            if len(aspect_terms.intersection(doc_terms)) >= len(aspect_terms) * 0.5:
                return True
        return False

    def _determine_uncovered_aspects(self, all_aspects, all_retrieved_docs):
        """Determine which aspects have not been adequately covered by retrieved documents."""
        uncovered = []
        for aspect in all_aspects:
            if not self._check_aspect_coverage(aspect, all_retrieved_docs):
                uncovered.append(aspect)
        return uncovered if uncovered else all_aspects  # Fallback to all if all seem covered

    def forward(self, claim):
        # Step 1: Identify key aspects of the claim
        aspect_result = self.aspect_identifier(claim=claim)
        all_aspects = aspect_result.aspects

        # Ensure we have 2-4 aspects
        if len(all_aspects) < 2:
            all_aspects = [claim, claim]  # Fallback: use claim twice
        elif len(all_aspects) > 4:
            all_aspects = all_aspects[:4]  # Limit to 4

        all_retrieved_docs = []
        covered_aspects = []

        # HOP 1: Use the first aspect or the full claim for initial retrieval
        hop1_docs = self.retrieve_k(claim).passages
        all_retrieved_docs.extend(hop1_docs)

        # Determine which aspects are covered after hop 1
        uncovered_aspects = self._determine_uncovered_aspects(all_aspects, all_retrieved_docs)

        # Mark covered aspects
        for aspect in all_aspects:
            if aspect not in uncovered_aspects:
                covered_aspects.append(aspect)

        # HOP 2: Target highest-priority uncovered aspect
        if uncovered_aspects:
            target_aspect_hop2 = uncovered_aspects[0]
        else:
            # If all covered, target the first aspect for deeper exploration
            target_aspect_hop2 = all_aspects[0]

        hop2_query = self.query_generator(
            claim=claim,
            target_aspect=target_aspect_hop2,
            covered_aspects=", ".join(covered_aspects) if covered_aspects else "None"
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        all_retrieved_docs.extend(hop2_docs)

        # Update coverage after hop 2
        uncovered_aspects = self._determine_uncovered_aspects(all_aspects, all_retrieved_docs)
        covered_aspects = [a for a in all_aspects if a not in uncovered_aspects]

        # HOP 3: Target next highest-priority uncovered aspect
        if uncovered_aspects:
            target_aspect_hop3 = uncovered_aspects[0]
        else:
            # If all covered, target a different aspect for broader exploration
            target_aspect_hop3 = all_aspects[1] if len(all_aspects) > 1 else all_aspects[0]

        hop3_query = self.query_generator(
            claim=claim,
            target_aspect=target_aspect_hop3,
            covered_aspects=", ".join(covered_aspects) if covered_aspects else "None"
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Return all 21 documents (7 per hop)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
