import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.k = 15  # Increased from 7 to 15 for diversity
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def _deduplicate_and_rerank(self, hop1_docs, hop2_docs, hop3_docs, top_k=21):
        """
        Post-processing module that deduplicates and reranks documents.

        Args:
            hop1_docs: Documents from first hop
            hop2_docs: Documents from second hop
            hop3_docs: Documents from third hop
            top_k: Number of documents to return (default: 21)

        Returns:
            List of top_k unique documents ordered by relevance score
        """
        # Dictionary to track document occurrences and positions
        # Key: document title, Value: dict with document, hop appearances, and positions
        doc_tracker = {}

        # Process documents from each hop
        for hop_idx, hop_docs in enumerate([hop1_docs, hop2_docs, hop3_docs], start=1):
            for position, doc in enumerate(hop_docs):
                # Extract document title (format: "title | content")
                title = doc.split(" | ")[0] if " | " in doc else doc

                if title not in doc_tracker:
                    doc_tracker[title] = {
                        'document': doc,
                        'hop_appearances': [],
                        'positions': []
                    }

                # Track which hop and position this document appeared in
                doc_tracker[title]['hop_appearances'].append(hop_idx)
                doc_tracker[title]['positions'].append((hop_idx, position))

        # Calculate relevance scores for each unique document
        scored_docs = []
        for title, info in doc_tracker.items():
            doc = info['document']
            hop_appearances = info['hop_appearances']
            positions = info['positions']

            # Cross-hop frequency score: documents appearing in multiple hops are more important
            # Score ranges from 1.0 (single hop) to 3.0 (all three hops)
            cross_hop_score = len(set(hop_appearances))

            # Position-based score: earlier positions in each hop are more relevant
            # We use reciprocal rank averaging across all appearances
            position_scores = []
            for hop_idx, position in positions:
                # Reciprocal rank: 1/(position+1), scaled by hop weight
                # Earlier hops get slightly higher weight (hop1: 1.0, hop2: 0.95, hop3: 0.9)
                hop_weight = 1.0 - (hop_idx - 1) * 0.05
                position_score = hop_weight / (position + 1)
                position_scores.append(position_score)

            avg_position_score = sum(position_scores) / len(position_scores)

            # Combined relevance score: balance cross-hop frequency and position ranking
            # Weight cross-hop frequency more heavily (60%) than position (40%)
            relevance_score = (0.6 * cross_hop_score) + (0.4 * avg_position_score * 10)

            scored_docs.append({
                'document': doc,
                'title': title,
                'score': relevance_score,
                'cross_hop_count': cross_hop_score,
                'avg_position_score': avg_position_score
            })

        # Sort by relevance score (descending) and return top_k documents
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # Return exactly top_k unique documents
        return [item['document'] for item in scored_docs[:top_k]]

    def forward(self, claim):
        # HOP 1: Retrieve 15 documents
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2: Retrieve 15 documents
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Retrieve 15 documents (total 45 documents retrieved)
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Post-processing: deduplicate and rerank to get top 21 unique documents
        final_docs = self._deduplicate_and_rerank(hop1_docs, hop2_docs, hop3_docs, top_k=21)

        return dspy.Prediction(retrieved_docs=final_docs)
