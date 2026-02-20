import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class AnalyzeClaimEntities(dspy.Signature):
    """Analyze the claim and extract key entities and concepts that need to be verified.
    Use chain of thought reasoning to identify the most important entities, people, places, events, or concepts mentioned in the claim."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    key_entities: list[str] = dspy.OutputField(desc="list of 2-3 key entities, concepts, or topics that are central to verifying this claim")
    reasoning: str = dspy.OutputField(desc="explanation of why these entities are important for verification")


class GenerateEntityQueries(dspy.Signature):
    """Generate 2-3 targeted search queries, one for each key entity or concept.
    Each query should be designed to retrieve documents specifically about that entity in the context of the claim."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    key_entities: list[str] = dspy.InputField(desc="key entities extracted from the claim")
    queries: list[str] = dspy.OutputField(desc="2-3 targeted search queries, one per entity/concept")


class RerankDocuments(dspy.Signature):
    """Rerank all retrieved documents based on their relevance to verifying the claim.
    Use chain of thought reasoning to assess which documents contain the most critical information for verification.
    Output relevance scores (0-100) for each document."""

    claim: str = dspy.InputField(desc="the claim to be verified")
    documents: list[str] = dspy.InputField(desc="all retrieved documents to rerank")
    relevance_scores: list[int] = dspy.OutputField(desc="relevance score (0-100) for each document, in the same order as input documents")
    reasoning: str = dspy.OutputField(desc="explanation of the reranking strategy and key factors considered")


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        # Stage 1: Entity extraction
        self.analyze_entities = dspy.ChainOfThought(AnalyzeClaimEntities)
        # Stage 2: Query generation
        self.generate_queries = dspy.Predict(GenerateEntityQueries)
        # Stage 3: Retrieval with k=50 per query
        self.retrieve_k50 = dspy.Retrieve(k=50)
        # Stage 4: Reranking
        self.rerank = dspy.ChainOfThought(RerankDocuments)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Extract key entities and concepts from the claim
            entity_analysis = self.analyze_entities(claim=claim)
            key_entities = entity_analysis.key_entities

            # Stage 2: Generate 2-3 targeted queries (one per entity)
            query_generation = self.generate_queries(
                claim=claim,
                key_entities=key_entities
            )
            queries = query_generation.queries

            # Ensure we have 2-3 queries (limit to 3 to meet search constraint)
            if len(queries) > 3:
                queries = queries[:3]
            elif len(queries) < 2:
                # Fallback: if fewer than 2 queries, add the claim itself
                queries = queries + [claim]
                queries = queries[:3]

            # Stage 3: Retrieve k=50 documents per query
            all_retrieved_docs = []
            for query in queries:
                docs = self.retrieve_k50(query).passages
                all_retrieved_docs.extend(docs)

            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                if doc not in seen:
                    seen.add(doc)
                    unique_docs.append(doc)

            # Stage 4: Rerank all documents (up to 150)
            # Limit to 150 documents if we have more (should be at most 150 from 3 queries * 50)
            docs_to_rerank = unique_docs[:150]

            rerank_result = self.rerank(
                claim=claim,
                documents=docs_to_rerank
            )
            relevance_scores = rerank_result.relevance_scores

            # Ensure we have a score for each document
            if len(relevance_scores) < len(docs_to_rerank):
                # Pad with zeros if needed
                relevance_scores = relevance_scores + [0] * (len(docs_to_rerank) - len(relevance_scores))
            elif len(relevance_scores) > len(docs_to_rerank):
                # Trim if we have too many scores
                relevance_scores = relevance_scores[:len(docs_to_rerank)]

            # Sort documents by relevance score (descending) and select top 21
            doc_score_pairs = list(zip(docs_to_rerank, relevance_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_21_docs = [doc for doc, score in doc_score_pairs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
