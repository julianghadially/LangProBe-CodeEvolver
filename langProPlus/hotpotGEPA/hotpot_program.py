import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class GenerateAnswer(dspy.Signature):
    """Answer questions with a short factoid answer."""

    question = dspy.InputField()
    hop1_passages = dspy.InputField()
    hop2_passages = dspy.InputField()
    answer = dspy.OutputField(desc="The answer itself and nothing else")

class ExtractFactoidAnswer(dspy.Signature):
    """Extract only the core factoid answer from a raw answer."""

    question = dspy.InputField()
    raw_answer = dspy.InputField()
    answer = dspy.OutputField(desc="only the core factoid answer without any extra descriptive text, parentheticals, or elaboration")

class RankPassageRelevance(dspy.Signature):
    """Score the relevance of a passage to a question on a scale of 1-10."""

    question = dspy.InputField()
    passage = dspy.InputField()
    relevance_score = dspy.OutputField(desc="A relevance score from 1 (not relevant) to 10 (highly relevant)")

class HotpotMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    """ChainOfThought variant with full passage reasoning."""

    def __init__(self):
        super().__init__()
        self.k = 7
        self.top_k_after_rerank = 3
        self.create_query_hop2 = dspy.Predict("question,hop1_passages->query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.rank_passage = dspy.ChainOfThought(RankPassageRelevance)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.extract_factoid = dspy.Predict(ExtractFactoidAnswer)

    def rerank_passages(self, question, passages):
        """Rerank passages using LLM-based relevance scoring and filter to top k."""
        scored_passages = []

        for passage in passages:
            # Score each passage using ChainOfThought reasoning
            result = self.rank_passage(question=question, passage=passage)
            try:
                # Extract numeric score from the relevance_score field
                score = float(result.relevance_score)
            except (ValueError, AttributeError):
                # If parsing fails, try to extract first number from string
                import re
                match = re.search(r'\d+', str(result.relevance_score))
                score = float(match.group()) if match else 5.0  # Default to middle score

            scored_passages.append((score, passage))

        # Sort by score (descending) and keep top k passages
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        top_passages = [passage for score, passage in scored_passages[:self.top_k_after_rerank]]

        return top_passages

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        # Rerank hop1 passages: filter from k=7 to top 3 most relevant
        hop1_docs_reranked = self.rerank_passages(question, hop1_docs)

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, hop1_passages=hop1_docs_reranked).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        # Rerank hop2 passages: filter from k=7 to top 3 most relevant
        hop2_docs_reranked = self.rerank_passages(hop2_query, hop2_docs)

        # Generate answer with Chain-of-Thought reasoning over reranked passages
        answer = self.generate_answer(
            question=question, hop1_passages=hop1_docs_reranked, hop2_passages=hop2_docs_reranked
        ).answer

        # Extract only the core factoid answer without extra descriptive text
        final_answer = self.extract_factoid(question=question, raw_answer=answer).answer

        return dspy.Prediction(answer=final_answer)
