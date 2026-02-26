import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# ============ New DSPy Signature Classes for Backward Chaining with Constraint Propagation ============

class HypothesisGenerator(dspy.Signature):
    """Generate 3-4 competing hypotheses about what types of supporting documents would prove or disprove the claim.
    Each hypothesis should describe a specific kind of evidence that would be decisive."""

    claim: str = dspy.InputField(desc="the claim to verify")
    hypotheses: list[str] = dspy.OutputField(desc="3-4 competing hypotheses about what evidence would prove/disprove the claim")


class ConstraintExtractor(dspy.Signature):
    """Extract specific constraints from the claim that documents must satisfy.
    Focus on concrete requirements: entity names, dates, relationships, negations, and specific factual requirements."""

    claim: str = dspy.InputField(desc="the claim to verify")
    constraints: list[str] = dspy.OutputField(desc="list of specific constraints that supporting documents must satisfy (e.g., 'Must mention entity X', 'Must describe relationship between X and Y', 'Must contain date information', 'Must confirm/deny specific fact')")


class BackwardQuery(dspy.Signature):
    """Generate a targeted search query designed to find documents that satisfy specific constraints and test a hypothesis.
    The query should be optimized to retrieve documents containing the required entities, relationships, or facts."""

    hypothesis: str = dspy.InputField(desc="the hypothesis being tested")
    constraints: str = dspy.InputField(desc="the constraints that documents must satisfy")
    claim: str = dspy.InputField(desc="the original claim for context")
    query: str = dspy.OutputField(desc="a targeted search query designed to find documents satisfying the constraints")


class ConstraintSatisfactionSignature(dspy.Signature):
    """Score how many constraints from the claim this document satisfies on a scale of 0-10.
    Count each constraint the document addresses: entities mentioned, relationships described, dates provided, facts confirmed/denied."""

    claim: str = dspy.InputField(desc="the claim being verified")
    constraints: str = dspy.InputField(desc="the constraints extracted from the claim")
    document: str = dspy.InputField(desc="the document to evaluate")
    reasoning: str = dspy.OutputField(desc="explanation of which constraints this document satisfies and which it doesn't")
    score: int = dspy.OutputField(desc="constraint satisfaction score from 0 (satisfies no constraints) to 10 (satisfies all key constraints)")


class ConstraintSatisfactionScorer(dspy.Module):
    """Module that scores documents by constraint satisfaction using chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ConstraintSatisfactionSignature)

    def forward(self, claim, constraints, document):
        return self.scorer(claim=claim, constraints=constraints, document=document)


# ============ Main Pipeline with Backward Chaining and Constraint Propagation ============

class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi-hop system for retrieving documents for a provided claim using Backward Chaining with Constraint Propagation.

    ARCHITECTURE:
    - Replaces forward claim decomposition with backward constraint satisfaction
    - Generates competing hypotheses about what evidence would prove/disprove the claim
    - Extracts specific constraints (entities, dates, relationships) from the claim
    - Performs backward search targeting documents that satisfy constraints
    - Scores documents by constraint satisfaction + diversity to avoid redundancy

    EVALUATION:
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize backward chaining modules
        self.hypothesis_generator = dspy.ChainOfThought(HypothesisGenerator)
        self.constraint_extractor = dspy.ChainOfThought(ConstraintExtractor)
        self.backward_query_generator = dspy.ChainOfThought(BackwardQuery)
        self.constraint_scorer = ConstraintSatisfactionScorer()

        # Retrieval module with k=7 for backward search
        self.retrieve_k = dspy.Retrieve(k=7)

    def _cosine_similarity_simple(self, text1, text2):
        """Simple word-based similarity for diversity scoring."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def _calculate_diversity_penalty(self, doc, selected_docs):
        """Calculate penalty based on similarity to already-selected documents."""
        if not selected_docs:
            return 0.0

        max_similarity = 0.0
        for selected_doc in selected_docs:
            similarity = self._cosine_similarity_simple(doc, selected_doc)
            max_similarity = max(max_similarity, similarity)

        # Penalty increases with similarity (0-3 range)
        return max_similarity * 3.0

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # ========== STEP 1: Generate Hypotheses and Extract Constraints ==========
            # Generate 3-4 competing hypotheses
            try:
                hypothesis_result = self.hypothesis_generator(claim=claim)
                hypotheses = hypothesis_result.hypotheses

                # Ensure hypotheses is a list
                if not isinstance(hypotheses, list):
                    if isinstance(hypotheses, str):
                        hypotheses = [h.strip() for h in hypotheses.split('\n') if h.strip()]
                        hypotheses = [h.lstrip('0123456789.-)> ').strip() for h in hypotheses if h.strip()]
                    else:
                        hypotheses = [str(hypotheses)]

                # Limit to 4 hypotheses
                hypotheses = hypotheses[:4]
            except Exception:
                # Fallback: create default hypothesis
                hypotheses = [f"Documents that directly discuss the entities and facts mentioned in: {claim}"]

            # Extract constraints from the claim
            try:
                constraint_result = self.constraint_extractor(claim=claim)
                constraints = constraint_result.constraints

                # Ensure constraints is a list
                if not isinstance(constraints, list):
                    if isinstance(constraints, str):
                        constraints = [c.strip() for c in constraints.split('\n') if c.strip()]
                        constraints = [c.lstrip('0123456789.-)> ').strip() for c in constraints if c.strip()]
                    else:
                        constraints = [str(constraints)]

            except Exception:
                # Fallback constraints
                constraints = ["Must be relevant to the claim"]

            # Format constraints as a string for scoring
            constraints_str = "\n".join(constraints)

            # ========== STEP 2: Backward Search with Constraint Targeting ==========
            # Select top 2-3 hypotheses to search (prioritize diversity)
            num_hypotheses_to_search = min(3, len(hypotheses))
            selected_hypotheses = hypotheses[:num_hypotheses_to_search]

            # Calculate how many searches per hypothesis (max 3 total searches)
            max_total_searches = 3
            searches_per_hypothesis = max_total_searches // num_hypotheses_to_search
            remaining_searches = max_total_searches % num_hypotheses_to_search

            all_retrieved_docs = []
            seen_titles = set()

            # Perform backward search for each selected hypothesis
            for idx, hypothesis in enumerate(selected_hypotheses):
                # Determine number of searches for this hypothesis
                num_searches = searches_per_hypothesis
                if idx < remaining_searches:
                    num_searches += 1

                # Generate backward queries for this hypothesis
                hypothesis_queries = []
                for _ in range(num_searches):
                    try:
                        query_result = self.backward_query_generator(
                            hypothesis=hypothesis,
                            constraints=constraints_str,
                            claim=claim
                        )
                        query = query_result.query
                        if query and isinstance(query, str):
                            hypothesis_queries.append(query)
                    except Exception:
                        # Fallback: use hypothesis as query
                        hypothesis_queries.append(hypothesis)

                # Retrieve documents for each query (k=7 per query)
                for query in hypothesis_queries[:num_searches]:
                    try:
                        docs = self.retrieve_k(query).passages

                        # Deduplicate by title as we add
                        for doc in docs:
                            title = doc.split(" | ")[0]
                            if title not in seen_titles:
                                seen_titles.add(title)
                                all_retrieved_docs.append(doc)
                    except Exception:
                        # If retrieval fails, continue
                        pass

            # ========== STEP 3: Score by Constraint Satisfaction + Diversity ==========
            scored_docs = []
            selected_docs_for_diversity = []

            for doc in all_retrieved_docs:
                try:
                    # Get constraint satisfaction score (0-10)
                    score_result = self.constraint_scorer(
                        claim=claim,
                        constraints=constraints_str,
                        document=doc
                    )

                    try:
                        constraint_score = int(score_result.score)
                    except (ValueError, TypeError):
                        constraint_score = 5

                    # Calculate diversity penalty (penalize similar documents)
                    diversity_penalty = self._calculate_diversity_penalty(doc, selected_docs_for_diversity)

                    # Final score: constraint satisfaction - diversity penalty
                    final_score = constraint_score - diversity_penalty

                    scored_docs.append((doc, final_score, constraint_score))

                except Exception:
                    # If scoring fails, assign neutral score
                    scored_docs.append((doc, 5.0, 5))

            # ========== STEP 4: Select Top 21 Documents ==========
            # Sort by final score (constraint satisfaction - diversity penalty)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Select top 21, updating diversity tracking as we go
            final_docs = []
            for doc, final_score, constraint_score in scored_docs:
                if len(final_docs) < 21:
                    final_docs.append(doc)
                    selected_docs_for_diversity.append(doc)

            return dspy.Prediction(retrieved_docs=final_docs)
