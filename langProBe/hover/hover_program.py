import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class QueryMutationGenerator(dspy.Signature):
    """Generate a mutated query variation based on a specific strategy."""
    claim: str = dspy.InputField()
    mutation_strategy: str = dspy.InputField(desc="focus strategy: entity-focused, temporal-focused, causal-focused, or relationship-focused")
    mutated_query: str = dspy.OutputField(desc="query variation based on strategy")


class QueryFitnessScorer(dspy.Signature):
    """Score a query based on quality of retrieved documents."""
    claim: str = dspy.InputField()
    query: str = dspy.InputField()
    passages: str = dspy.InputField(desc="retrieved passages as concatenated string")
    fitness_score: float = dspy.OutputField(desc="0-1 score based on diversity, relevance, and coverage")
    reasoning: str = dspy.OutputField(desc="explanation of the score")


class QueryCrossover(dspy.Signature):
    """Combine features of two high-performing queries into an offspring query."""
    claim: str = dspy.InputField()
    query1: str = dspy.InputField(desc="first parent query")
    query2: str = dspy.InputField(desc="second parent query")
    fitness1: str = dspy.InputField(desc="fitness reasoning for query1")
    fitness2: str = dspy.InputField(desc="fitness reasoning for query2")
    offspring_query: str = dspy.OutputField(desc="combined query with best features of both parents")


class RelevanceReranker(dspy.Signature):
    """Rerank passages by relevance to the claim."""
    claim: str = dspy.InputField()
    passages: str = dspy.InputField(desc="all retrieved passages as concatenated string")
    ranked_indices: str = dspy.OutputField(desc="comma-separated indices of top 21 most relevant passages (0-indexed)")


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.mutation_generator = dspy.Predict(QueryMutationGenerator)
        self.fitness_scorer = dspy.Predict(QueryFitnessScorer)
        self.crossover = dspy.Predict(QueryCrossover)
        self.reranker = dspy.Predict(RelevanceReranker)

    def forward(self, claim):
        # Stage 1: Query Mutation Generation
        # Generate 4 diverse query mutations using different strategies in one forward pass
        strategies = ["entity-focused", "temporal-focused", "causal-focused", "relationship-focused"]
        mutations = []

        for strategy in strategies:
            mutation_result = self.mutation_generator(claim=claim, mutation_strategy=strategy)
            mutated_query = mutation_result.mutated_query
            mutations.append(mutated_query)

        # Stage 2: Fitness Evaluation & Evolution
        # For each mutation, retrieve k=15 documents and evaluate fitness
        fitness_results = []

        for mutation_query in mutations:
            # Retrieve k=15 documents for this mutation
            retrieve_mutation = dspy.Retrieve(k=15)
            mutation_docs = retrieve_mutation(mutation_query).passages

            # Convert passages to string for the fitness scorer
            passages_str = "\n\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(mutation_docs)])

            # Score this query's fitness
            fitness_result = self.fitness_scorer(
                claim=claim,
                query=mutation_query,
                passages=passages_str
            )

            fitness_results.append({
                'query': mutation_query,
                'score': float(fitness_result.fitness_score) if isinstance(fitness_result.fitness_score, (int, float)) else 0.5,
                'reasoning': fitness_result.reasoning
            })

        # Sort by fitness score and select top 2
        fitness_results.sort(key=lambda x: x['score'], reverse=True)
        top_queries = fitness_results[:2]

        # Generate offspring query through crossover
        offspring_result = self.crossover(
            claim=claim,
            query1=top_queries[0]['query'],
            query2=top_queries[1]['query'],
            fitness1=top_queries[0]['reasoning'],
            fitness2=top_queries[1]['reasoning']
        )
        offspring_query = offspring_result.offspring_query

        # Stage 3: Final Retrieval with Reranking
        # Use the offspring query to retrieve k=63 documents
        retrieve_final = dspy.Retrieve(k=63)
        final_docs = retrieve_final(offspring_query).passages

        # Rerank all 63 documents to get top 21
        passages_str = "\n\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(final_docs)])
        rerank_result = self.reranker(claim=claim, passages=passages_str)

        # Parse the ranked indices
        try:
            ranked_indices_str = rerank_result.ranked_indices
            # Handle various formats: "0,1,2,..." or "0, 1, 2, ..." or even list-like formats
            ranked_indices_str = ranked_indices_str.strip().strip('[]')
            ranked_indices = [int(idx.strip()) for idx in ranked_indices_str.split(',')]

            # Take top 21 and ensure they're valid
            ranked_indices = [idx for idx in ranked_indices[:21] if 0 <= idx < len(final_docs)]

            # If we don't have 21 valid indices, pad with remaining documents
            if len(ranked_indices) < 21:
                remaining = [i for i in range(len(final_docs)) if i not in ranked_indices]
                ranked_indices.extend(remaining[:21 - len(ranked_indices)])

            # Reorder documents according to ranked indices
            top_21_docs = [final_docs[idx] for idx in ranked_indices[:21]]
        except (ValueError, IndexError, AttributeError):
            # Fallback: just take the first 21 documents if parsing fails
            top_21_docs = final_docs[:21]

        return dspy.Prediction(retrieved_docs=top_21_docs)
