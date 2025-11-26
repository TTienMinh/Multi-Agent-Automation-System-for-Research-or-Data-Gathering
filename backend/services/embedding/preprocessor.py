import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter


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
    Chunk the given text into smaller pieces.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    if text is None:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks