import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 10  # Retrieve 10 documents per hop for 30 total

        # Entity chain extraction signature
        self.entity_chain_extractor = dspy.ChainOfThought("claim -> entity_chain")

        # Retrieve module
        self.retrieve_k = dspy.Retrieve(k=self.k)

        # Relevance scoring for reranking
        self.score_relevance = dspy.ChainOfThought("claim, document -> relevance_score")

    def forward(self, claim):
        # Step 1: Extract entity chain (2-4 key entities)
        entity_chain_result = self.entity_chain_extractor(claim=claim)
        entity_chain = entity_chain_result.entity_chain

        # Parse entity chain (assuming it returns a list or comma-separated string)
        if isinstance(entity_chain, str):
            entities = [e.strip() for e in entity_chain.split(',')]
        else:
            entities = entity_chain

        # Ensure we have at least 3 entities for 3 hops (pad if needed)
        while len(entities) < 3:
            entities.append(claim)

        # Step 2: HOP 1 - Retrieve using first entity
        hop1_query = entities[0]
        hop1_docs = self.retrieve_k(hop1_query).passages
        hop1_context = " ".join(hop1_docs[:3])  # Use top 3 docs as context

        # Step 3: HOP 2 - Retrieve using second entity + context from hop 1
        hop2_query = f"{entities[1]} {hop1_context[:200]}"  # Include limited context
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_context = " ".join(hop2_docs[:3])

        # Step 4: HOP 3 - Retrieve using remaining entities + context from hops 1-2
        remaining_entities = " ".join(entities[2:])
        combined_context = f"{hop1_context[:100]} {hop2_context[:100]}"
        hop3_query = f"{remaining_entities} {combined_context}"
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Step 5: Combine all documents (30 total)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Step 6: Rerank documents by relevance score
        doc_scores = []
        for doc in all_docs:
            try:
                score_result = self.score_relevance(claim=claim, document=doc)
                # Extract relevance score (handle different formats)
                if hasattr(score_result, 'relevance_score'):
                    score = float(score_result.relevance_score)
                else:
                    # Try to extract numeric value from string
                    score_str = str(score_result).strip()
                    score = float(score_str) if score_str.replace('.', '').isdigit() else 0.5
            except (ValueError, AttributeError):
                score = 0.5  # Default score if parsing fails

            doc_scores.append((doc, score))

        # Sort by relevance score (descending) and keep top 21
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_21_docs = [doc for doc, score in doc_scores[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
