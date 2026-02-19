import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from typing import List

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


# DSPy Signature for extracting claim aspects
class ClaimAspectExtractor(dspy.Signature):
    """Analyze the claim and extract 3 distinct retrieval perspectives to find supporting documents.

    The 3 perspectives should be:
    1. Primary entities: key people, places, organizations, or concepts mentioned
    2. Relationships/connections: how entities relate to each other, actions, events, or interactions
    3. Contextual/temporal info: time periods, circumstances, conditions, or background context
    """

    claim: str = dspy.InputField(desc="the claim to analyze")
    primary_entities: str = dspy.OutputField(desc="primary entities, people, places, organizations, or key concepts in the claim")
    relationships: str = dspy.OutputField(desc="relationships, connections, actions, events, or interactions between entities")
    contextual_info: str = dspy.OutputField(desc="contextual, temporal, or background information needed to verify the claim")


# DSPy Signature for generating specialized queries per aspect
class AspectQueryGenerator(dspy.Signature):
    """Generate a specialized search query based on a specific aspect of the claim.

    The query should be optimized to retrieve documents that provide evidence about this particular aspect.
    """

    claim: str = dspy.InputField(desc="the original claim to verify")
    aspect_type: str = dspy.InputField(desc="the type of aspect (entities, relationships, or context)")
    aspect_description: str = dspy.InputField(desc="description of this specific aspect to focus on")
    query: str = dspy.OutputField(desc="a specialized search query optimized for retrieving documents about this aspect")


# DSPy Signature for cross-attention reranking
class CrossAttentionReranker(dspy.Signature):
    """Analyze all retrieved documents grouped by aspect and select exactly 21 documents that provide the most comprehensive and complementary evidence.

    Use chain-of-thought reasoning to:
    1. Identify which documents are most relevant to the claim
    2. Ensure diversity across all three aspects (entities, relationships, context)
    3. Check that documents provide coherent, non-redundant information
    4. Select documents that together form a complete evidence base

    Output exactly 21 document indices (0-74) with reasoning for each selection.
    """

    claim: str = dspy.InputField(desc="the claim being verified")
    entities_docs: List[str] = dspy.InputField(desc="25 documents retrieved for primary entities aspect (indices 0-24)")
    relationships_docs: List[str] = dspy.InputField(desc="25 documents retrieved for relationships aspect (indices 25-49)")
    context_docs: List[str] = dspy.InputField(desc="25 documents retrieved for contextual info aspect (indices 50-74)")
    reasoning: str = dspy.OutputField(desc="chain-of-thought reasoning about document selection strategy, relevance, diversity, and coherence")
    selected_indices: List[int] = dspy.OutputField(desc="exactly 21 document indices (0-74) selected based on relevance, diversity, and coherence")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim using Cross-Attention Multi-Path Retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Extracts 3 retrieval perspectives from the claim (entities, relationships, context)
    - Performs 3 parallel specialized searches with k=25 documents each (total 75 docs)
    - Uses cross-attention reranking to select 21 most complementary documents
    '''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)

        # Initialize DSPy modules for the cross-attention multi-path retrieval
        self.aspect_extractor = dspy.ChainOfThought(ClaimAspectExtractor)
        self.query_generator = dspy.Predict(AspectQueryGenerator)
        self.reranker = dspy.ChainOfThought(CrossAttentionReranker)
        self.retrieve_k25 = dspy.Retrieve(k=25)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Step 1: Extract 3 distinct retrieval perspectives from the claim
            aspects = self.aspect_extractor(claim=claim)

            # Step 2: Generate specialized queries for each aspect and retrieve k=25 documents in parallel
            # These will be executed in parallel as separate dspy.Retrieve calls

            # Query for primary entities aspect
            entities_query = self.query_generator(
                claim=claim,
                aspect_type="primary entities",
                aspect_description=aspects.primary_entities
            ).query
            entities_docs = self.retrieve_k25(entities_query).passages

            # Query for relationships/connections aspect
            relationships_query = self.query_generator(
                claim=claim,
                aspect_type="relationships and connections",
                aspect_description=aspects.relationships
            ).query
            relationships_docs = self.retrieve_k25(relationships_query).passages

            # Query for contextual/temporal info aspect
            context_query = self.query_generator(
                claim=claim,
                aspect_type="contextual and temporal information",
                aspect_description=aspects.contextual_info
            ).query
            context_docs = self.retrieve_k25(context_query).passages

            # Step 3: Use cross-attention reranker to select exactly 21 documents
            # The reranker considers relevance, diversity across aspects, and cross-document coherence
            reranking_result = self.reranker(
                claim=claim,
                entities_docs=entities_docs,
                relationships_docs=relationships_docs,
                context_docs=context_docs
            )

            # Step 4: Combine all documents and select the reranked subset
            all_docs = entities_docs + relationships_docs + context_docs
            selected_indices = reranking_result.selected_indices

            # Ensure we have exactly 21 unique indices
            # Handle potential issues with LLM output
            if not isinstance(selected_indices, list):
                selected_indices = list(range(21))  # fallback to first 21 docs

            # Deduplicate and validate indices
            valid_indices = []
            seen = set()
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < 75 and idx not in seen:
                    valid_indices.append(idx)
                    seen.add(idx)

            # If we don't have enough valid indices, fill from beginning
            if len(valid_indices) < 21:
                for i in range(75):
                    if i not in seen and len(valid_indices) < 21:
                        valid_indices.append(i)
                        seen.add(i)

            # Take exactly 21 documents
            final_indices = valid_indices[:21]
            selected_docs = [all_docs[i] for i in final_indices]

            return dspy.Prediction(retrieved_docs=selected_docs)
