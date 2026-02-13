import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class DocumentRelevanceScorer(dspy.Signature):
    """Score the relevance of a document to a claim on a scale of 1-10 with reasoning."""

    claim = dspy.InputField(desc="The claim to verify")
    document = dspy.InputField(desc="The retrieved document to score")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning for the relevance score")
    score = dspy.OutputField(desc="Relevance score from 1-10, where 10 is highly relevant")


class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 15  # Increased from 9 to 15 for retrieval
        self.top_k = 21  # Final number of documents to return after reranking
        self.extract_key_terms = dspy.Predict("claim->key_terms")
        self.create_query_hop2 = dspy.Predict("claim,key_terms,hop1_titles->query")
        self.create_query_hop3 = dspy.Predict("claim,key_terms,hop1_titles,hop2_titles->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.score_relevance = dspy.ChainOfThought(DocumentRelevanceScorer)

    def forward(self, claim):
        # Extract key terms from claim (3-5 key named entities or concepts)
        key_terms = self.extract_key_terms(claim=claim).key_terms

        # HOP 1: Retrieve 15 documents
        hop1_docs = self.retrieve_k(claim).passages
        hop1_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100] for doc in hop1_docs])

        # HOP 2: Retrieve 15 documents
        hop2_query = self.create_query_hop2(
            claim=claim, key_terms=key_terms, hop1_titles=hop1_titles
        ).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        hop2_titles = ", ".join([doc.split("\n")[0] if "\n" in doc else doc[:100] for doc in hop2_docs])

        # HOP 3: Retrieve 15 documents
        hop3_query = self.create_query_hop3(
            claim=claim, key_terms=key_terms, hop1_titles=hop1_titles, hop2_titles=hop2_titles
        ).query
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all 45 retrieved documents
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # RERANKING STAGE: Score each document's relevance to the claim
        scored_docs = []
        for doc in all_docs:
            try:
                result = self.score_relevance(claim=claim, document=doc)
                # Extract numeric score from the output
                score_str = result.score.strip()
                # Handle various score formats (e.g., "8", "8/10", "8 out of 10")
                if '/' in score_str:
                    score = float(score_str.split('/')[0])
                elif 'out of' in score_str.lower():
                    score = float(score_str.split()[0])
                else:
                    score = float(score_str)
                scored_docs.append((doc, score, result.reasoning))
            except (ValueError, AttributeError):
                # If scoring fails, assign a default score of 5 (neutral)
                scored_docs.append((doc, 5.0, "Scoring failed, assigned neutral score"))

        # Sort documents by relevance score (descending) and take top 21
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score, reasoning in scored_docs[:self.top_k]]

        return dspy.Prediction(retrieved_docs=top_docs)


