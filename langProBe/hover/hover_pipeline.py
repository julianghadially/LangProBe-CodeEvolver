import dspy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langProBe.dspy_program import LangProBeDSPyMetaProgram

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class DiversityReranker(dspy.Module):
    """Reranks retrieved documents to maximize diversity using clustering.

    Takes k=50 over-retrieved documents, clusters them into semantic groups,
    and selects the top-scored document from each of the 7 most relevant clusters.
    """

    def __init__(self, target_k=7):
        super().__init__()
        self.target_k = target_k
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def forward(self, passages, scores=None):
        """
        Args:
            passages: List of passage strings (length ~50)
            scores: Optional list of relevance scores from retriever

        Returns:
            List of top-k diverse passages (length=target_k)
        """
        if len(passages) <= self.target_k:
            return passages

        # Generate embeddings for all passages
        embeddings = self.embedding_model.encode(passages, convert_to_numpy=True)

        # Cluster into target_k clusters
        kmeans = KMeans(n_clusters=self.target_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # If scores not provided, use uniform scores
        if scores is None:
            scores = [1.0] * len(passages)

        # Select best document from each cluster
        diverse_passages = []
        for cluster_id in range(self.target_k):
            # Get indices of passages in this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

            if cluster_indices:
                # Select passage with highest score in this cluster
                best_idx = max(cluster_indices, key=lambda i: scores[i])
                diverse_passages.append(passages[best_idx])

        # If some clusters are empty, fill with top remaining documents
        while len(diverse_passages) < self.target_k:
            # Find passages not yet selected
            selected_set = set(diverse_passages)
            remaining = [(i, p, scores[i]) for i, p in enumerate(passages) if p not in selected_set]
            if remaining:
                # Add highest scoring remaining passage
                _, best_passage, _ = max(remaining, key=lambda x: x[2])
                diverse_passages.append(best_passage)
            else:
                break

        return diverse_passages[:self.target_k]


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with diversity-aware reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self, use_diversity=True):
        super().__init__()
        self.use_diversity = use_diversity
        self.over_retrieve_k = 50 if use_diversity else 7
        self.final_k = 7

        # Initialize diversity reranker
        if use_diversity:
            self.reranker = DiversityReranker(target_k=self.final_k)

        # Query generation modules
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

        # Retrieval module (over-retrieves if diversity enabled)
        self.retrieve_k = dspy.Retrieve(k=self.over_retrieve_k)

        # Summarization modules
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1: Retrieve and rerank for diversity
        hop1_retrieval = self.retrieve_k(claim)
        hop1_docs_raw = hop1_retrieval.passages

        if self.use_diversity:
            # Extract scores if available (ColBERT provides scores)
            scores = getattr(hop1_retrieval, 'scores', None)
            hop1_docs = self.reranker(passages=hop1_docs_raw, scores=scores)
        else:
            hop1_docs = hop1_docs_raw[:self.final_k]

        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary

        # HOP 2: Retrieve and rerank for diversity
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_retrieval = self.retrieve_k(hop2_query)
        hop2_docs_raw = hop2_retrieval.passages

        if self.use_diversity:
            scores = getattr(hop2_retrieval, 'scores', None)
            hop2_docs = self.reranker(passages=hop2_docs_raw, scores=scores)
        else:
            hop2_docs = hop2_docs_raw[:self.final_k]

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: Retrieve and rerank for diversity
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_retrieval = self.retrieve_k(hop3_query)
        hop3_docs_raw = hop3_retrieval.passages

        if self.use_diversity:
            scores = getattr(hop3_retrieval, 'scores', None)
            hop3_docs = self.reranker(passages=hop3_docs_raw, scores=scores)
        else:
            hop3_docs = hop3_docs_raw[:self.final_k]

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop(use_diversity=True)

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            return self.program(claim=claim)
