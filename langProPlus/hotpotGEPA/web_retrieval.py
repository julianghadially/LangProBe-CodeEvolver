"""Custom Wikipedia retrieval module using Serper search and Firecrawl scraping."""

import dspy
from dspy.predict.parameter import Parameter
from services import SerperService, FirecrawlService, SearchResult, ScrapedPage
import random
import re


class PassageObject:
    """Simple wrapper to make passages compatible with DSPy's expected format."""
    def __init__(self, text: str):
        self.long_text = text


class WikipediaWebRetrieval(Parameter):
    """Custom retrieval that searches Wikipedia via Serper and scrapes via Firecrawl.

    This module replaces the ColBERTv2 retrieval system with a web search + scraping
    pipeline that retrieves full Wikipedia article content instead of abstracts.

    Attributes:
        k: Number of passages to return per query (default 7).
        max_content_length: Maximum characters to scrape from a page (default 10000).
        serper: SerperService instance for Wikipedia searches.
        firecrawl: FirecrawlService instance for page scraping.
    """

    def __init__(self, k=7, max_content_length=10000):
        """Initialize the Wikipedia web retrieval module.

        Args:
            k: Number of passages to return per query.
            max_content_length: Maximum characters to scrape from pages.
        """
        super().__init__()
        self.k = k
        self.max_content_length = max_content_length
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()

        # DSPy Parameter interface requirements
        self.stage = random.randbytes(8).hex()
        self.name = "WikipediaWebSearch"
        self.input_variable = "query"
        self.callbacks = []

    def __call__(self, query: str, k: int | None = None, **kwargs):
        """Make the retrieval module callable (required by DSPy).

        Returns a list of PassageObject instances with .long_text attributes,
        which is what dspy.Retrieve expects when calling dspy.settings.rm().
        """
        k = k if k is not None else self.k

        # Get passages as strings
        prediction = self.forward(query, k=k, **kwargs)

        # Wrap in PassageObject for DSPy compatibility
        return [PassageObject(passage) for passage in prediction.passages]

    def forward(self, query: str, k: int | None = None, **kwargs) -> dspy.Prediction:
        """Search Wikipedia, scrape top result, and chunk into passages.

        This method implements the core retrieval pipeline:
        1. Search Wikipedia via Serper
        2. Extract Wikipedia URLs from search results
        3. Scrape the top result via Firecrawl (constraint: 1 page per query)
        4. Chunk scraped markdown into k passages
        5. Fallback to search snippets if scraping fails

        Args:
            query: Search query string.
            k: Number of passages to return (overrides self.k if provided).
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            dspy.Prediction with passages attribute containing list of k strings.
        """
        k = k if k is not None else self.k

        # Step 1: Search Wikipedia via Serper
        search_results = self._search_wikipedia(query)

        # Step 2: Extract Wikipedia URLs (top 3-5)
        wiki_urls = self._extract_wikipedia_urls(search_results)

        # Step 3: Scrape the TOP result only (constraint: 1 page per query)
        scraped_content = self._scrape_page(wiki_urls[0]) if wiki_urls else None

        # Step 4: Chunk scraped markdown into k passages
        passages = self._chunk_into_passages(scraped_content, k)

        # Step 5: Fallback to search snippets if scraping failed
        if not passages or len(passages) < k:
            passages = self._fallback_to_snippets(search_results, k)

        return dspy.Prediction(passages=passages)

    def _search_wikipedia(self, query: str) -> list[SearchResult]:
        """Search Wikipedia using Serper with site:wikipedia.org filter.

        Args:
            query: Search query string.

        Returns:
            List of SearchResult objects from Serper.
        """
        wikipedia_query = f"{query} site:wikipedia.org"
        return self.serper.search(wikipedia_query, num_results=10)

    def _extract_wikipedia_urls(self, results: list[SearchResult]) -> list[str]:
        """Extract Wikipedia URLs from search results.

        Args:
            results: List of SearchResult objects.

        Returns:
            List of Wikipedia URLs (top 3-5).
        """
        wiki_urls = []
        for result in results:
            if "wikipedia.org" in result.link.lower():
                wiki_urls.append(result.link)
                if len(wiki_urls) >= 5:  # Top 3-5 URLs
                    break

        if not wiki_urls:
            print(f"Warning: No Wikipedia URLs found in search results")

        return wiki_urls

    def _scrape_page(self, url: str) -> ScrapedPage | None:
        """Scrape a single Wikipedia page using Firecrawl.

        CRITICAL: Only scrapes ONE page per query to comply with constraints.

        Args:
            url: Wikipedia page URL to scrape.

        Returns:
            ScrapedPage object if successful, None if scraping fails.
        """
        try:
            scraped = self.firecrawl.scrape(url, max_length=self.max_content_length)
            if not scraped.success:
                print(f"Warning: Scraping failed for {url}: {scraped.error}")
                return None
            return scraped
        except Exception as e:
            print(f"Error: Scraping exception for {url}: {e}")
            return None

    def _chunk_into_passages(self, scraped: ScrapedPage | None, k: int) -> list[str]:
        """Chunk scraped markdown content into k passages.

        Applies chunking strategies in order of preference:
        1. Split by markdown headers (## or ###) - preserves Wikipedia sections
        2. Split by double newlines (paragraphs) - preserves coherence
        3. Split by character count (600-800 chars) - ensures k passages

        Args:
            scraped: ScrapedPage object with markdown content.
            k: Target number of passages to return.

        Returns:
            List of exactly k passage strings (padded with empty strings if needed).
        """
        if not scraped or not scraped.markdown:
            return []

        markdown = scraped.markdown
        passages = []

        # Strategy 1: Split by markdown headers (Wikipedia sections)
        sections = self._split_by_headers(markdown)
        if len(sections) >= k:
            # Take first k sections
            passages = sections[:k]
        else:
            # Strategy 2: Split sections into paragraphs
            for section in sections:
                paragraphs = self._split_by_paragraphs(section)
                passages.extend(paragraphs)
                if len(passages) >= k:
                    break

        # Strategy 3: If still not enough, chunk by character count
        if len(passages) < k:
            passages = self._split_by_chars(markdown, target_count=k)

        # Ensure we return exactly k passages
        if len(passages) > k:
            passages = passages[:k]
        elif len(passages) < k:
            # Pad with empty strings
            passages.extend([""] * (k - len(passages)))

        return passages

    def _split_by_headers(self, markdown: str) -> list[str]:
        """Split markdown by ## or ### headers (Wikipedia sections).

        Args:
            markdown: Markdown content string.

        Returns:
            List of section strings.
        """
        # Match ## Header or ### Header (but not # single)
        sections = re.split(r'\n(?=##+ )', markdown)
        return [s.strip() for s in sections if s.strip()]

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by double newlines (paragraphs).

        Args:
            text: Text content string.

        Returns:
            List of paragraph strings (minimum 50 characters).
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

    def _split_by_chars(self, text: str, target_count: int, chunk_size: int = 600) -> list[str]:
        """Split text into roughly equal chunks of chunk_size characters.

        Args:
            text: Text content string.
            target_count: Target number of chunks to create.
            chunk_size: Approximate characters per chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
            if len(chunks) >= target_count:
                break
        return chunks

    def _fallback_to_snippets(self, results: list[SearchResult], k: int) -> list[str]:
        """Fallback: Use search result snippets as passages.

        This is called when scraping fails or returns insufficient content.

        Args:
            results: List of SearchResult objects.
            k: Number of passages needed.

        Returns:
            List of exactly k snippet strings (padded with empty strings if needed).
        """
        passages = [result.snippet for result in results[:k]]
        # Pad if necessary
        while len(passages) < k:
            passages.append("")
        return passages[:k]

    # DSPy Parameter interface methods

    def reset(self):
        """Reset module state (required by Parameter interface)."""
        pass

    def dump_state(self):
        """Dump module state for serialization.

        Returns:
            Dictionary with module configuration.
        """
        return {"k": self.k, "max_content_length": self.max_content_length}

    def load_state(self, state):
        """Load module state from serialization.

        Args:
            state: Dictionary with module configuration.
        """
        for name, value in state.items():
            setattr(self, name, value)
