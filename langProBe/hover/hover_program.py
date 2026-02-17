import dspy
from langProBe.dspy_program import LangProBeDSPyMetaProgram

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    '''Multi hop system for retrieving documents for a provided claim. 
    
    EVALUATION
    - This system is assessed by retrieving the correct documents that are most relevant. 
    - The system must provide at most 21 documents at the end of the program.'''

    def __init__(self):
        super().__init__()
        # Adaptive k values for each hop (60 total documents retrieved)
        self.k_hop1 = 25
        self.k_hop2 = 20
        self.k_hop3 = 15
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_hop1 = dspy.Retrieve(k=self.k_hop1)
        self.retrieve_hop2 = dspy.Retrieve(k=self.k_hop2)
        self.retrieve_hop3 = dspy.Retrieve(k=self.k_hop3)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_hop1(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # Extract titles from hop 1
        hop1_titles = set()
        for doc in hop1_docs:
            # Extract title from document (assuming format "title | text" or similar)
            title = self._extract_title(doc)
            hop1_titles.add(title)

        # Track all raw retrievals for frequency counting
        all_raw_docs = list(hop1_docs)

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs_raw = self.retrieve_hop2(hop2_query).passages
        all_raw_docs.extend(hop2_docs_raw)

        # Deduplicate hop 2 documents
        hop2_docs = []
        hop2_titles = set()
        for doc in hop2_docs_raw:
            title = self._extract_title(doc)
            if title not in hop1_titles:
                hop2_docs.append(doc)
                hop2_titles.add(title)

        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs_raw = self.retrieve_hop3(hop3_query).passages
        all_raw_docs.extend(hop3_docs_raw)

        # Deduplicate hop 3 documents
        hop3_docs = []
        combined_titles = hop1_titles | hop2_titles
        for doc in hop3_docs_raw:
            title = self._extract_title(doc)
            if title not in combined_titles:
                hop3_docs.append(doc)

        # Combine all deduplicated documents
        all_deduplicated_docs = list(hop1_docs) + hop2_docs + hop3_docs

        # Title-based reranking: count frequency across all raw retrievals
        title_frequency = {}
        for doc in all_raw_docs:
            title = self._extract_title(doc)
            title_frequency[title] = title_frequency.get(title, 0) + 1

        # Sort deduplicated documents by frequency (descending)
        reranked_docs = sorted(
            all_deduplicated_docs,
            key=lambda doc: title_frequency.get(self._extract_title(doc), 0),
            reverse=True
        )

        # Return exactly 21 documents
        final_docs = reranked_docs[:21]

        return dspy.Prediction(retrieved_docs=final_docs)

    def _extract_title(self, doc):
        """Extract title from document string.

        Assumes document format is either:
        - "title | text" (pipe-separated)
        - Just the text (use first 100 chars as unique identifier)
        """
        if isinstance(doc, str):
            # Try to split by pipe
            if ' | ' in doc:
                return doc.split(' | ')[0].strip()
            # Fallback: use first 100 characters as identifier
            return doc[:100].strip()
        # If doc is an object with a title attribute
        elif hasattr(doc, 'title'):
            return doc.title
        # Fallback: convert to string
        return str(doc)[:100].strip()
