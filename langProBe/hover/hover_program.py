import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.entity_decomposer = dspy.Predict("claim->entities, queries")
        self.retrieve_k = dspy.Retrieve(k=self.k)

    def forward(self, claim):
        # Entity-aware query decomposition
        decomposition = self.entity_decomposer(claim=claim)
        entities = decomposition.entities
        queries = decomposition.queries

        # Parse queries into a list (assuming comma-separated or newline-separated)
        if isinstance(queries, str):
            query_list = [q.strip() for q in queries.replace('\n', ',').split(',') if q.strip()]
        else:
            query_list = queries if isinstance(queries, list) else [queries]

        # Distribute queries across 3 hops
        num_queries = len(query_list)
        if num_queries == 0:
            # Fallback: use original claim if no queries generated
            query_list = [claim]
            num_queries = 1

        # Calculate distribution across hops (e.g., 5 queries -> 2-2-1, 3 queries -> 1-1-1)
        hop1_count = (num_queries + 2) // 3  # Round up for first hop
        hop2_count = (num_queries + 1) // 3  # Middle hop
        hop3_count = num_queries // 3  # Remaining for last hop

        # Adjust if counts don't sum to total (edge cases)
        while hop1_count + hop2_count + hop3_count < num_queries:
            if hop3_count < hop1_count:
                hop3_count += 1
            elif hop2_count < hop1_count:
                hop2_count += 1
            else:
                hop1_count += 1

        # HOP 1: Use first set of queries
        hop1_queries = query_list[:hop1_count]
        hop1_docs = []
        for query in hop1_queries:
            hop1_docs.extend(self.retrieve_k(query).passages)

        # HOP 2: Use second set of queries
        hop2_start = hop1_count
        hop2_end = hop1_count + hop2_count
        hop2_queries = query_list[hop2_start:hop2_end] if hop2_end > hop2_start else [claim]
        hop2_docs = []
        for query in hop2_queries:
            hop2_docs.extend(self.retrieve_k(query).passages)

        # HOP 3: Use remaining queries
        hop3_start = hop2_end
        hop3_queries = query_list[hop3_start:] if hop3_start < num_queries else [claim]
        hop3_docs = []
        for query in hop3_queries:
            hop3_docs.extend(self.retrieve_k(query).passages)

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


