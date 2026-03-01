import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HopChainExtractor(dspy.Signature):
    """Analyze the claim to identify the logical hop structure and entity bridges needed for multi-hop reasoning.

    A hop is a logical step in the reasoning chain. For example:
    - Hop 1: Identify the concrete entity (e.g., book title, organization name)
    - Hop 2: Find the bridge entity (e.g., author of the book, founder of organization)
    - Hop 3: Verify final fact about the bridge entity (e.g., birth date, nationality)

    Extract the hop chain structure showing what information is needed at each step."""

    claim: str = dspy.InputField(desc="The claim to analyze for multi-hop reasoning structure")
    hop_chain: str = dspy.OutputField(desc="The logical hop structure as a numbered list (e.g., 'Hop 1: title -> author | Hop 2: author -> birth details')")
    concrete_entities: str = dspy.OutputField(desc="The most concrete/specific entities mentioned in the claim that should be retrieved first")


class HopTargetedQuery(dspy.Signature):
    """Generate a single highly-focused query for a specific missing hop or bridge entity.

    Based on the claim, hop chain structure, and already-retrieved documents, generate ONE specific query that:
    - Targets the next missing hop in the reasoning chain
    - Uses information from previously retrieved documents as context
    - Focuses on bridging entities that connect different hops

    Generate exactly ONE query."""

    claim: str = dspy.InputField(desc="The claim being verified")
    hop_chain: str = dspy.InputField(desc="The logical hop structure extracted from the claim")
    retrieved_context: str = dspy.InputField(desc="Documents already retrieved in previous hops")
    missing_hop: str = dspy.InputField(desc="The specific hop or bridge entity that needs to be retrieved")
    query: str = dspy.OutputField(desc="A single highly-focused query targeting the missing hop")


class ChainCompletenessEvaluator(dspy.Signature):
    """Evaluate whether all hops in the reasoning chain have bridging documents retrieved.

    For each hop in the chain, assess whether:
    - The hop has relevant documents that provide the needed information
    - Bridge entities connecting hops are present
    - The chain of reasoning is complete from start to finish

    Output a completeness assessment and identify the next missing hop if incomplete."""

    claim: str = dspy.InputField(desc="The claim being verified")
    hop_chain: str = dspy.InputField(desc="The logical hop structure extracted from the claim")
    retrieved_documents: str = dspy.InputField(desc="All documents retrieved so far")
    is_complete: str = dspy.OutputField(desc="'yes' if all hops are covered by retrieved documents, 'no' if gaps remain")
    missing_hop: str = dspy.OutputField(desc="Description of the next missing hop or bridge entity, or 'none' if complete")


class EntityAndGapAnalyzer(dspy.Signature):
    """Analyze the claim to extract multiple entity chains and identify information gaps that need to be verified.

    Entity chains are sequences of related entities mentioned in the claim (e.g., person -> organization -> location).
    Information gaps are missing pieces of information needed to verify the claim.
    Generate 2-3 distinct search queries targeting different entity chains or information gaps."""

    claim: str = dspy.InputField(desc="The claim to analyze for entities and information gaps")
    entity_chains: str = dspy.OutputField(desc="2-3 distinct entity chains or key topics extracted from the claim, separated by newlines")
    queries: list[str] = dspy.OutputField(desc="2-3 parallel search queries targeting different entity chains or information gaps (must be 2-3 queries)")


class DocumentRelevanceScorer(dspy.Signature):
    """Score each document's relevance to the claim and identified entity chains.

    Consider:
    - How well the document addresses the claim
    - Coverage of mentioned entity chains
    - Factual information that helps verify the claim

    Return a relevance score from 0-100 for each document."""

    claim: str = dspy.InputField(desc="The original claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    document: str = dspy.InputField(desc="A retrieved document to score")
    relevance_score: int = dspy.OutputField(desc="Relevance score from 0-100 indicating how relevant this document is to verifying the claim")


class ConfidenceEvaluator(dspy.Signature):
    """Assess whether retrieved documents provide sufficient evidence to verify the claim.

    Analyze the coverage of entity chains and facts in the retrieved documents:
    - Identify which entity chains or facts are well-covered
    - Identify which entity chains, bridging entities, or facts are missing or have insufficient coverage
    - Consider indirect references and incomplete entity chain coverage

    Output a confidence score (0-100) indicating sufficiency of evidence and list specific information gaps."""

    claim: str = dspy.InputField(desc="The claim being verified")
    entity_chains: str = dspy.InputField(desc="The entity chains extracted from the claim")
    retrieved_documents: str = dspy.InputField(desc="The documents retrieved in round 1, concatenated")
    confidence_score: int = dspy.OutputField(desc="Confidence score from 0-100 indicating whether documents provide sufficient evidence to verify the claim")
    missing_information: str = dspy.OutputField(desc="Specific information gaps, missing entity chains, or bridging entities that need to be retrieved")


class TargetedQueryGenerator(dspy.Signature):
    """Generate 1-2 highly targeted follow-up queries to address missing information gaps.

    Based on the claim, already-retrieved documents, and identified gaps, generate specific queries that:
    - Target missing bridging entities or entity chains
    - Address indirect references not covered in round 1
    - Fill specific factual gaps identified in the confidence evaluation

    Generate between 1-2 queries maximum."""

    claim: str = dspy.InputField(desc="The claim being verified")
    retrieved_documents: str = dspy.InputField(desc="Documents already retrieved in round 1")
    missing_information: str = dspy.InputField(desc="Specific information gaps that need to be addressed")
    targeted_queries: list[str] = dspy.OutputField(desc="1-2 highly targeted follow-up queries addressing the missing information (must be 1-2 queries)")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using sequential hop-by-hop reasoning.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Sequential hop retrieval configuration
        self.k_hop1 = 25  # Retrieve more documents for the first hop (concrete entities)
        self.k_hop_followup = 20  # Retrieve documents for follow-up hops
        self.max_final_docs = 21  # Final output constraint
        self.max_hops = 3  # Maximum number of hops (to stay within query limit)

        # Hop chain extraction module
        self.hop_extractor = dspy.ChainOfThought(HopChainExtractor)

        # Initial retrieval for first hop
        self.retrieve_hop1 = dspy.Retrieve(k=self.k_hop1)

        # Follow-up retrieval for subsequent hops
        self.retrieve_followup = dspy.Retrieve(k=self.k_hop_followup)

        # Chain completeness evaluator
        self.completeness_evaluator = dspy.ChainOfThought(ChainCompletenessEvaluator)

        # Hop-targeted query generator
        self.hop_query_generator = dspy.ChainOfThought(HopTargetedQuery)

        # Document relevance scorer (kept for backward compatibility)
        self.doc_scorer = dspy.ChainOfThought(DocumentRelevanceScorer)

    def forward(self, claim):
        # ===== STEP 1: Extract hop chain structure =====
        # Analyze the claim to identify the logical hop structure and entity bridges
        try:
            hop_analysis = self.hop_extractor(claim=claim)
            hop_chain = hop_analysis.hop_chain
            concrete_entities = hop_analysis.concrete_entities
        except Exception:
            # Fallback if hop extraction fails
            hop_chain = "Hop 1: Identify entities | Hop 2: Verify relationships | Hop 3: Confirm facts"
            concrete_entities = claim

        # ===== STEP 2: Retrieve documents for first hop (concrete entities) =====
        # Start with the most concrete entities mentioned in the claim
        hop1_docs = self.retrieve_hop1(concrete_entities).passages

        # Track all retrieved documents and their hop assignments for position-aware reranking
        all_docs = []
        doc_hop_map = {}  # Maps doc -> set of hop numbers it covers
        seen = set()

        for doc in hop1_docs:
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)
                doc_hop_map[doc] = {1}  # Mark as covering hop 1

        # ===== STEP 3: Iterative hop-by-hop retrieval =====
        # Retrieve documents for remaining hops (max 2 additional hops to stay within 3 query limit)
        current_hop = 1

        while current_hop < self.max_hops:
            # Evaluate chain completeness
            retrieved_docs_str = "\n\n".join(all_docs[:50])  # Limit to 50 docs to avoid token limits

            try:
                completeness_eval = self.completeness_evaluator(
                    claim=claim,
                    hop_chain=hop_chain,
                    retrieved_documents=retrieved_docs_str
                )
                is_complete = completeness_eval.is_complete.lower().strip()
                missing_hop = completeness_eval.missing_hop

                # If chain is complete, stop retrieval
                if is_complete == 'yes' or missing_hop.lower().strip() == 'none':
                    break

            except Exception:
                # If evaluation fails, assume we need more hops (conservative approach)
                missing_hop = f"Hop {current_hop + 1} information"

            # ===== STEP 4: Generate targeted query for missing hop =====
            try:
                hop_query_result = self.hop_query_generator(
                    claim=claim,
                    hop_chain=hop_chain,
                    retrieved_context=retrieved_docs_str,
                    missing_hop=missing_hop
                )
                hop_query = hop_query_result.query
            except Exception:
                # Fallback query if generation fails
                hop_query = claim

            # ===== STEP 5: Retrieve documents for this hop =====
            hop_docs = self.retrieve_followup(hop_query).passages

            # Add new documents and track their hop coverage
            for doc in hop_docs:
                if doc not in seen:
                    seen.add(doc)
                    all_docs.append(doc)
                    doc_hop_map[doc] = {current_hop + 1}
                else:
                    # Document appears in multiple hops - it's a bridge document
                    if doc in doc_hop_map:
                        doc_hop_map[doc].add(current_hop + 1)

            current_hop += 1

        # ===== STEP 6: Position-aware reranking =====
        # Prioritize documents covering multiple hops (bridge documents)
        if len(all_docs) <= self.max_final_docs:
            return dspy.Prediction(retrieved_docs=all_docs)

        # Calculate position-aware scores
        doc_scores = []
        for i, doc in enumerate(all_docs):
            # Base score: inverse position (earlier = higher score)
            position_score = len(all_docs) - i

            # Multi-hop bonus: documents covering multiple hops get priority
            hop_coverage = len(doc_hop_map.get(doc, {1}))
            multi_hop_bonus = hop_coverage * 100

            # Early hop bonus: documents from hop 1 get slight priority (concrete entities)
            early_hop_bonus = 50 if 1 in doc_hop_map.get(doc, set()) else 0

            # Combined score
            total_score = position_score + multi_hop_bonus + early_hop_bonus
            doc_scores.append((doc, total_score))

        # Sort by score descending and take top max_final_docs
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in doc_scores[:self.max_final_docs]]

        return dspy.Prediction(retrieved_docs=top_docs)
