import re


def clean_html(html_text: str) -> str:
    """
    Clean HTML tags from the given text.

    Args:
        html_text (str): Text containing HTML tags.

    Returns: 
        str: Cleaned text without HTML tags.
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', html_text)
    return cleantext


def extract_text_from_markdown(markdown_text: str) -> str:
    """
    Extract plain text from markdown formatted text.

    Args:
        markdown_text (str): Text in markdown format.

    Returns:
        str: Extracted plain text.
    """
    # Remove markdown syntax (e.g., headers, lists)
    cleanr = re.compile(r'\!\[.*?\]\(.*?\)|\*\*.*?\*\*|__.*?__|`.*?`|>\s.*?(\n|$)|\n\s*\n', re.DOTALL)
    cleantext = re.sub(cleanr, '', markdown_text)
    return cleantext


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Chunk text into smaller pieces with specified size and overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks