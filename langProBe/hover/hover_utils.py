import dspy


def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))


def extract_doc_title(doc: str) -> str:
    """
    Extract title from document string.

    Documents are formatted as "Title | Content". This function extracts
    the title portion for use in deduplication and evaluation.

    Args:
        doc: Document string in "Title | Content" format

    Returns:
        The title portion of the document (before " | ")
    """
    return doc.split(" | ")[0] if " | " in doc else doc


def format_doc_snippet(doc: str, max_length: int = 200) -> str:
    """
    Format document with truncated content for display.

    This is useful for logging, debugging, or displaying documents in a
    compact format while preserving the title and some context.

    Args:
        doc: Document string in "Title | Content" format
        max_length: Maximum length of content snippet (default: 200)

    Returns:
        Formatted string with truncated content: "Title | Content[0:max_length]..."
    """
    parts = doc.split(" | ", 1)
    if len(parts) == 2:
        title, content = parts
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return f"{title} | {content}"
    return doc[:max_length] + "..." if len(doc) > max_length else doc


# Constraint: Do NOT return more than 21 documents for evaluation.
MAX_RETRIEVED_DOCS = 21


def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs[:MAX_RETRIEVED_DOCS]],
        )
    )
    return gold_titles.issubset(found_titles)
