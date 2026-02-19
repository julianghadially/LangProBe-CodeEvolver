import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from .hover_program import HoverMultiHop

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"


class SentenceRelevanceSignature(dspy.Signature):
    """Extract and score the 2-3 most relevant sentences from a document that support or refute the given claim."""

    claim: str = dspy.InputField(desc="the claim to verify")
    document_text: str = dspy.InputField(desc="the full document text to analyze")
    relevance_score: float = dspy.OutputField(desc="relevance score from 0.0 to 1.0 indicating how relevant the document is to the claim")
    best_sentences: str = dspy.OutputField(desc="the 2-3 most relevant sentences from the document, separated by spaces")


class SentenceRelevanceScorer(dspy.Module):
    """DSPy module that uses LLM to extract and score the most relevant sentences from a document."""

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(SentenceRelevanceSignature)

    def forward(self, claim, document_text):
        result = self.scorer(claim=claim, document_text=document_text)
        return result


class HoverMultiHopPipeline(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim with two-stage sentence-level retrieval.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Stage 1: Perform 3-hop retrieval with k=50 documents per query (total 150 docs)
    - Stage 2: Use SentenceRelevanceScorer to extract and score top sentences from each document
    - Re-rank all passages by relevance score and return top 21 as "title | extracted_sentences"
    '''

    def __init__(self):
        super().__init__()
        self.rm = dspy.ColBERTv2(url=COLBERT_URL)
        self.program = HoverMultiHop()
        self.sentence_scorer = SentenceRelevanceScorer()
        # Override the k value for stage 1 retrieval
        self.k_stage1 = 50

    def forward(self, claim):
        with dspy.context(rm=self.rm):
            # Stage 1: Retrieve k=50 documents per query using 3-hop retrieval
            # We need to modify the retrieval to get 50 docs instead of 7

            # HOP 1
            retrieve_50 = dspy.Retrieve(k=self.k_stage1)
            hop1_docs = retrieve_50(claim).passages
            summary_1 = self.program.summarize1(
                claim=claim, passages=hop1_docs[:7]  # Use first 7 for summary
            ).summary

            # HOP 2
            hop2_query = self.program.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = retrieve_50(hop2_query).passages
            summary_2 = self.program.summarize2(
                claim=claim, context=summary_1, passages=hop2_docs[:7]  # Use first 7 for summary
            ).summary

            # HOP 3
            hop3_query = self.program.create_query_hop3(
                claim=claim, summary_1=summary_1, summary_2=summary_2
            ).query
            hop3_docs = retrieve_50(hop3_query).passages

            # Combine all retrieved documents (up to 150 total)
            all_docs = hop1_docs + hop2_docs + hop3_docs

            # Stage 2: Extract and score sentences from each document
            scored_docs = []
            for doc in all_docs:
                # Document format is typically "title | text"
                if " | " in doc:
                    title, text = doc.split(" | ", 1)
                else:
                    # If no separator, treat whole thing as text with empty title
                    title = ""
                    text = doc

                # Use the sentence scorer to extract relevant sentences and get score
                try:
                    result = self.sentence_scorer(claim=claim, document_text=text)
                    relevance_score = float(result.relevance_score)
                    best_sentences = result.best_sentences

                    # Store the scored document with extracted sentences
                    scored_docs.append({
                        'title': title,
                        'sentences': best_sentences,
                        'score': relevance_score,
                        'formatted': f"{title} | {best_sentences}" if title else best_sentences
                    })
                except Exception as e:
                    # If scoring fails, use a low score and keep original text
                    scored_docs.append({
                        'title': title,
                        'sentences': text[:500],  # Truncate if needed
                        'score': 0.0,
                        'formatted': doc
                    })

            # Sort by relevance score (descending) and take top 21
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            top_21_docs = [doc['formatted'] for doc in scored_docs[:21]]

            return dspy.Prediction(retrieved_docs=top_21_docs)
