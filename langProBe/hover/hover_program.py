import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class VerifyClaimSupport(dspy.Signature):
    """Analyze the claim and retrieved documents to identify which specific supporting facts are still missing or unclear.

    Focus on identifying concrete factual gaps that would be needed to fully verify or refute the claim.
    List specific missing information rather than general categories."""

    claim = dspy.InputField(desc="The claim to verify")
    retrieved_documents = dspy.InputField(desc="All 21 documents retrieved so far")
    missing_facts = dspy.OutputField(desc="Specific supporting facts that are still missing or unclear from the documents")

class TargetedRetrievalQuery(dspy.Signature):
    """Generate a highly specific retrieval query focused solely on finding the missing information identified.

    The query should be laser-focused on filling the specific gaps rather than being broad.
    Use precise terms and entities mentioned in the missing facts."""

    claim = dspy.InputField(desc="The original claim being verified")
    missing_facts = dspy.InputField(desc="The specific missing or unclear supporting facts")
    targeted_query = dspy.OutputField(desc="A highly specific query to retrieve documents addressing the missing facts")

class DocumentRelevanceScorer(dspy.Signature):
    """Score how well a document fills the identified gaps in supporting the claim.

    Focus on whether the document provides the specific missing facts identified in the verification step.
    Score from 0-10 where 10 means the document directly addresses multiple missing facts."""

    claim = dspy.InputField(desc="The claim being verified")
    missing_facts = dspy.InputField(desc="The specific missing supporting facts")
    document = dspy.InputField(desc="A single document to score")
    relevance_score = dspy.OutputField(desc="Relevance score from 0-10 based on how well it fills identified gaps")
    reasoning = dspy.OutputField(desc="Brief explanation of the score")

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.retrieve_targeted = dspy.Retrieve(k=21)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        # Verification-guided retrieval components
        self.verify_claim_support = dspy.ChainOfThought(VerifyClaimSupport)
        self.generate_targeted_query = dspy.ChainOfThought(TargetedRetrievalQuery)
        self.score_document = dspy.ChainOfThought(DocumentRelevanceScorer)

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all initial documents (21 total: 7+7+7)
        initial_docs = hop1_docs + hop2_docs + hop3_docs

        # VERIFICATION-GUIDED RETRIEVAL LOOP
        # Step 1: Verify what supporting facts are missing
        verification_result = self.verify_claim_support(
            claim=claim,
            retrieved_documents="\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(initial_docs)])
        )
        missing_facts = verification_result.missing_facts

        # Step 2: Generate targeted query for missing information
        targeted_query_result = self.generate_targeted_query(
            claim=claim,
            missing_facts=missing_facts
        )
        targeted_query = targeted_query_result.targeted_query

        # Step 3: Execute targeted retrieval with k=21
        targeted_docs = self.retrieve_targeted(targeted_query).passages

        # Step 4: Score all targeted documents based on gap-filling capability
        targeted_scores = []
        for doc in targeted_docs:
            score_result = self.score_document(
                claim=claim,
                missing_facts=missing_facts,
                document=doc
            )
            try:
                # Extract numeric score from the output
                score = float(score_result.relevance_score)
            except (ValueError, AttributeError):
                # If parsing fails, assign a default low score
                score = 0.0
            targeted_scores.append((doc, score))

        # Sort targeted docs by score (highest first)
        targeted_scores.sort(key=lambda x: x[1], reverse=True)
        top_targeted_docs = [doc for doc, score in targeted_scores[:7]]

        # Step 5: Score initial documents to identify weakest ones
        initial_scores = []
        for doc in initial_docs:
            score_result = self.score_document(
                claim=claim,
                missing_facts=missing_facts,
                document=doc
            )
            try:
                score = float(score_result.relevance_score)
            except (ValueError, AttributeError):
                score = 5.0  # Default mid-range score for existing docs
            initial_scores.append((doc, score))

        # Sort initial docs by score (lowest first to identify weakest)
        initial_scores.sort(key=lambda x: x[1])

        # Step 6: Replace lowest-scoring 7 documents with highest-scoring 7 from targeted retrieval
        # Keep the top 14 from initial retrieval, add top 7 from targeted retrieval
        refined_docs = [doc for doc, score in initial_scores[7:]] + top_targeted_docs

        return dspy.Prediction(retrieved_docs=refined_docs)
