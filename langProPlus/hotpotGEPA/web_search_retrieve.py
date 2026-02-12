"""
Web search and scraping based retrieval module for DSPy.

Replaces dspy.Retrieve with a custom module that uses:
- SerperService for web search
- FirecrawlService for web page scraping
- Semantic markdown chunking for passage extraction
"""

import logging
import re
from typing import Optional

import dspy
from dspy import Prediction

logger = logging.getLogger(__name__)


def chunk_markdown(
    markdown: str, chunk_size: int = 700, preserve_headers: bool = True
) -> list[str]:
    """Split markdown into semantic chunks while preserving context.

    Args:
        markdown: Markdown content to chunk
        chunk_size: Target size for each chunk in characters (default 700)
        preserve_headers: Whether to prepend section headers to chunks (default True)

    Returns:
        List of markdown chunks suitable for passage reranking
    """
    if not markdown or len(markdown.strip()) == 0:
        return []

    chunks = []

    # Split by headers (##, ###, ####)
    # Pattern captures headers and splits content
    sections = re.split(r"\n(#{2,4}\s+.+)\n", markdown)

    current_header = ""
    i = 0
    while i < len(sections):
        section = sections[i].strip()

        # Check if this is a header
        if re.match(r"^#{2,4}\s+", section):
            current_header = section
            i += 1
            continue

        # Skip empty sections
        if not section:
            i += 1
            continue

        # Process content section
        if len(section) <= chunk_size:
            # Section fits in one chunk
            chunk_content = (
                f"{current_header}\n\n{section}" if preserve_headers and current_header else section
            )
            chunks.append(chunk_content.strip())
        else:
            # Split large section by paragraphs (double newlines)
            paragraphs = section.split("\n\n")
            current_chunk = f"{current_header}\n\n" if preserve_headers and current_header else ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if adding this paragraph exceeds chunk_size
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    # Finish current chunk if it has content
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    # Start new chunk with header (if preserving) and current paragraph
                    if preserve_headers and current_header:
                        current_chunk = f"{current_header}\n\n{para}\n\n"
                    else:
                        current_chunk = f"{para}\n\n"

            # Add remaining content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        i += 1

    # Fallback: if no chunks created, split by simple paragraphs
    if not chunks and markdown.strip():
        paragraphs = markdown.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = f"{para}\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    # Final fallback: if still no chunks, return whole text as single chunk
    if not chunks and markdown.strip():
        chunks.append(markdown.strip())

    return chunks


class WebSearchRetrieve(dspy.Retrieve):
    """Custom retrieval module using web search and scraping.

    Replaces dspy.Retrieve to provide web-based passage retrieval using:
    - SerperService for Google Search (returns search results with snippets)
    - FirecrawlService for web page scraping (returns full page markdown)
    - Semantic markdown chunking to create passage-like content

    The module maintains compatibility with dspy.Retrieve interface by returning
    Prediction(passages=[...]) which can be used as a drop-in replacement.
    """

    def __init__(
        self,
        k: int = 7,
        num_search_results: int = 5,
        scrape_top_n: int = 1,
        max_scrape_length: int = 10000,
        chunk_size: int = 700,
        include_snippets: bool = True,
        callbacks=None,
    ):
        """Initialize WebSearchRetrieve.

        Args:
            k: Number of passages to return (default 7, matching dspy.Retrieve)
            num_search_results: Number of search results to fetch (default 5)
            scrape_top_n: Number of top URLs to scrape (default 1)
            max_scrape_length: Maximum characters per scraped page (default 10000)
            chunk_size: Target size for markdown chunks in characters (default 700)
            include_snippets: Whether to include search snippets as passages (default True)
            callbacks: Optional callbacks for DSPy integration
        """
        super().__init__(k=k, callbacks=callbacks)
        self.num_search_results = num_search_results
        self.scrape_top_n = scrape_top_n
        self.max_scrape_length = max_scrape_length
        self.chunk_size = chunk_size
        self.include_snippets = include_snippets

        # Import services here to avoid early initialization
        from services import SerperService, FirecrawlService

        # Initialize services
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()

        logger.info(
            f"WebSearchRetrieve initialized: k={k}, num_search_results={num_search_results}, "
            f"scrape_top_n={scrape_top_n}, chunk_size={chunk_size}, include_snippets={include_snippets}"
        )

    def forward(self, query: str, k: Optional[int] = None, **kwargs) -> Prediction:
        """Execute web search and scraping to retrieve passages.

        Args:
            query: The search query
            k: Optional override for number of passages to return
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Prediction object with passages attribute containing list of passage strings
        """
        k = k if k is not None else self.k
        passages = []

        try:
            # Step 1: Search with SerperService
            logger.info(f"WebSearchRetrieve: Searching for query='{query}'")
            search_results = self.serper.search(
                query=query, num_results=self.num_search_results
            )

            if not search_results:
                logger.warning(f"No search results found for query: '{query}'")
                return Prediction(passages=[])

            logger.info(f"Found {len(search_results)} search results")

            # Step 2: Extract snippets as passages (if enabled)
            if self.include_snippets:
                for result in search_results:
                    snippet_passage = (
                        f"[Search Result {result.position}] {result.title}\n{result.snippet}"
                    )
                    passages.append(snippet_passage)
                logger.info(f"Added {len(search_results)} snippets as passages")

            # Step 3: Scrape top URL(s) and chunk markdown
            scraped_chunks = []
            scrape_success = False

            for i in range(min(self.scrape_top_n, len(search_results))):
                url = search_results[i].link
                logger.info(f"Attempting to scrape URL [{i+1}/{self.scrape_top_n}]: {url}")

                try:
                    scraped_page = self.firecrawl.scrape(
                        url=url, max_length=self.max_scrape_length
                    )

                    # Check if scraping succeeded and content is substantial
                    if scraped_page.success and len(scraped_page.markdown) > 100:
                        logger.info(
                            f"Successfully scraped {len(scraped_page.markdown)} chars from {url}"
                        )

                        # Chunk the markdown
                        chunks = chunk_markdown(
                            scraped_page.markdown,
                            chunk_size=self.chunk_size,
                            preserve_headers=True,
                        )

                        if chunks:
                            scraped_chunks.extend(chunks)
                            scrape_success = True
                            logger.info(f"Created {len(chunks)} chunks from scraped content")
                            break  # Successfully scraped, stop trying more URLs
                        else:
                            logger.warning(f"Chunking produced no results for {url}")
                    else:
                        error_msg = scraped_page.error if hasattr(scraped_page, 'error') else "Unknown error"
                        logger.warning(
                            f"Scraping failed for {url}: success={scraped_page.success}, "
                            f"content_length={len(scraped_page.markdown)}, error={error_msg}"
                        )

                except Exception as e:
                    logger.error(f"Exception while scraping {url}: {e}")
                    continue

            # Step 4: Combine passages (snippets first, then scrape chunks)
            passages.extend(scraped_chunks)

            # Step 5: Limit to k passages
            if len(passages) > k:
                passages = passages[:k]

            logger.info(
                f"WebSearchRetrieve complete: query='{query}', total_passages={len(passages)}, "
                f"scrape_success={scrape_success}"
            )

        except Exception as e:
            logger.error(f"WebSearchRetrieve failed for query '{query}': {e}")
            passages = []  # Return empty, let downstream components handle

        return Prediction(passages=passages)

    def __call__(self, *args, **kwargs):
        """Make the module callable like dspy.Retrieve."""
        return self.forward(*args, **kwargs)
