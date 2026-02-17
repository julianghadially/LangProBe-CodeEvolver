import dspy
import math
import re
from collections import Counter
from langProBe.dspy_program import LangProBeDSPyMetaProgram


class BM25Reranker:
    """Lightweight BM25 scorer for reranking documents against a claim."""

    def __init__(self):
        # BM25 parameters (standard values)
        self.k1 = 1.5  # term frequency saturation parameter
        self.b = 0.75  # length normalization parameter

    def tokenize(self, text):
        """Simple tokenization: lowercase, split on whitespace, remove punctuation."""
        # Lowercase
        text = text.lower()
        # Remove punctuation and split on whitespace
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def compute_idf(self, documents):
        """Compute inverse document frequency for each term."""
        # Count number of documents containing each term
        doc_count = len(documents)
        term_doc_count = Counter()

        for doc in documents:
            doc_tokens = set(self.tokenize(doc))
            for term in doc_tokens:
                term_doc_count[term] += 1

        # Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf_scores = {}
        for term, df in term_doc_count.items():
            idf_scores[term] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

        return idf_scores

    def compute_bm25_score(self, query_tokens, doc_tokens, avg_doc_len, idf_scores):
        """Compute BM25 score for a single document."""
        # Count term frequencies in document
        doc_term_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        # Compute BM25 score
        score = 0.0
        for term in query_tokens:
            if term not in idf_scores:
                continue

            idf = idf_scores[term]
            tf = doc_term_freq.get(term, 0)

            # BM25 formula: IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def rerank(self, claim, documents):
        """Rerank documents by BM25 score against the claim."""
        if not documents:
            return []

        # Tokenize claim
        query_tokens = self.tokenize(claim)

        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]

        # Compute IDF scores
        idf_scores = self.compute_idf(documents)

        # Calculate average document length
        avg_doc_len = sum(len(doc_tokens) for doc_tokens in tokenized_docs) / len(tokenized_docs)

        # Score each document
        scored_docs = []
        for doc, doc_tokens in zip(documents, tokenized_docs):
            score = self.compute_bm25_score(query_tokens, doc_tokens, avg_doc_len, idf_scores)
            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs


class EntityExtractorSignature(dspy.Signature):
    """Extract key entities, concepts, or information gaps that are NOT well-covered
    by the provided documents when trying to verify the claim. Focus on identifying
    missing pieces of information that would help verify the claim."""

    claim = dspy.InputField(desc="The claim to verify")
    documents = dspy.InputField(desc="Top documents retrieved so far")
    uncovered_entities = dspy.OutputField(desc="List of entities/concepts not yet covered, described as a comma-separated string")


class EntityExtractor(dspy.Module):
    """Extract uncovered entities from current document set."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EntityExtractorSignature)

    def forward(self, claim, documents):
        # Join top N documents (e.g., top 5) to avoid context overflow
        doc_context = "\n\n".join(documents[:5])
        result = self.extract(claim=claim, documents=doc_context)
        return result.uncovered_entities


class FocusedQueryGeneratorSignature(dspy.Signature):
    """Generate a focused search query targeting specific uncovered entities or information gaps.
    The query should be designed to retrieve documents that fill these gaps."""

    claim = dspy.InputField(desc="The original claim to verify")
    uncovered_entities = dspy.InputField(desc="Entities/concepts that need more coverage")
    search_query = dspy.OutputField(desc="Focused search query targeting these gaps")


class FocusedQueryGenerator(dspy.Module):
    """Generate a focused query for the next retrieval hop."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FocusedQueryGeneratorSignature)

    def forward(self, claim, uncovered_entities):
        result = self.generate(claim=claim, uncovered_entities=uncovered_entities)
        return result.search_query


class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Sequential multi-hop retrieval system with BM25 reranking.

    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant.
    - The system must provide at most 21 documents at the end of the program.

    ARCHITECTURE
    - Hop 1: Retrieve k=35 documents for the original claim
    - Hop 2: Extract uncovered entities, generate focused query, retrieve k=35 more
    - Hop 3: Identify remaining gaps, generate query, retrieve k=35 final documents
    - Deduplicate ~105 documents by normalized title
    - Rerank unique documents using BM25 scoring against claim
    - Return top 21 documents
    '''

    def __init__(self):
        super().__init__()
        self.k = 35  # Retrieve 35 documents per hop
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.entity_extractor = EntityExtractor()
        self.query_generator = FocusedQueryGenerator()
        self.bm25_reranker = BM25Reranker()

    def forward(self, claim):
        # HOP 1: Initial retrieval with the original claim
        hop1_docs = self.retrieve_k(claim).passages

        # HOP 2: Extract uncovered entities and retrieve more
        uncovered_entities = self.entity_extractor(claim=claim, documents=hop1_docs)
        hop2_query = self.query_generator(claim=claim, uncovered_entities=uncovered_entities)
        hop2_docs = self.retrieve_k(hop2_query).passages

        # HOP 3: Identify remaining gaps and retrieve final set
        combined_docs_so_far = hop1_docs + hop2_docs
        remaining_gaps = self.entity_extractor(claim=claim, documents=combined_docs_so_far)
        hop3_query = self.query_generator(claim=claim, uncovered_entities=remaining_gaps)
        hop3_docs = self.retrieve_k(hop3_query).passages

        # Combine all documents (3 × 35 = ~105 documents)
        all_docs = hop1_docs + hop2_docs + hop3_docs

        # Deduplicate by normalized title (format: "title | content")
        seen_titles = set()
        unique_docs = []
        for doc in all_docs:
            # Extract title using same format as evaluation (split on " | ")
            title = doc.split(" | ")[0] if " | " in doc else doc[:100]
            normalized_title = dspy.evaluate.normalize_text(title)

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_docs.append(doc)

        # Rerank unique documents using BM25
        ranked_docs = self.bm25_reranker.rerank(claim=claim, documents=unique_docs)

        # Return top 21 documents
        top_21_docs = [doc for doc, score in ranked_docs[:21]]

        return dspy.Prediction(retrieved_docs=top_21_docs)
