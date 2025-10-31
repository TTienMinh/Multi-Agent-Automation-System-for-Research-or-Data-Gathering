import os
import time
import arxiv
import dotenv
import logging
from typing import List
from pathlib import Path
from datetime import datetime
from theguardian import theguardian_content, theguardian_section

dotenv.load_dotenv()

project_root = Path(__file__).parent.parent.parent.parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_arxiv_abstracts(num_results: int = 10, query: str = "LLM") -> List[str]:
    """
    Fetch abstracts from ArXiv using the arxiv library.

    Args:
        num_results (int): Number of results to fetch.
        query (str): Search query for ArXiv.
    Returns:
        List[str]: List of formatted paper entries.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=num_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    try:
        all_results = list(client.results(search))
    except Exception as e:
        logging.error(f"Error fetching results from arXiv: {e}")
        return []

    if not all_results:
        logging.warning("No results found.")
        return []

    logging.info(f"Found and retrieved {len(all_results)} paper(s).")
    output_lines = []
    for i, paper in enumerate(all_results):
        abstract_clean = paper.summary.replace('\n', ' ')
        entry = (
            f"Paper {i+1}/{len(all_results)}:\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join([a.name for a in paper.authors])}\n"
            f"ID: {paper.entry_id}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"Abstract:\n{abstract_clean}\n"
            "--------------------------------------------------\n"
        )
        output_lines.append(entry)
        # Log a concise version
        logging.info(f"{i+1}. {paper.title} | Published: {paper.published.strftime('%Y-%m-%d')}")
    return output_lines


def save_entries_to_file(entries: List[str], filename: str) -> None:
    """
    Save a list of entries to a file in the data directory.

    Args:
        entries (List[str]): List of string entries to save.
        filename (str): Name of the file to save entries to.
    """
    full_path = project_root / "data" / filename
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(entries)
        logging.info(f"Successfully stored all data in {full_path}")
    except IOError as e:
        logging.error(f"Error writing to file: {e}")
        

def fetch_theguardian_sections(
    api_key: str,
    query: str = "AI OR artificial intelligence",
    tag: str = "technology/technology",
    from_date: str = "2025-05-01",
    order_by: str = "relevance"
) -> List[str]:
    """
    Fetch sections from The Guardian via API.

    Args:
        api_key (str): The Guardian API key.
        query (str): Search query.
        tag (str): Tag to filter content.
        from_date (str): Start date for articles (YYYY-MM-DD). Defaults to today if None.
        order_by (str): Order of results.
    Returns:
        List[str]: List of formatted section entries.
    """
    if from_date is None:
        from_date = datetime.now().strftime('%Y-%m-%d')
    headers = {
        "q": query,
        "tag": tag,
        "from-date": from_date,
        "order-by": order_by,
    }
    try:
        content = theguardian_content.Content(api=api_key, **headers)
        res = content.get_content_response()
        result = content.get_results(res)
    except Exception as e:
        logging.error(f"Error fetching results from The Guardian: {e}")
        return []

    output_lines = []
    for i, section in enumerate(result):
        entry = (
            f"Paper No. {i+1}/{len(result)}:\n"
            f"ID: {section.get('id', 'N/A')}\n"
            f"Section ID: {section.get('sectionId', 'N/A')}\n"
            f"Section Name: {section.get('sectionName', 'N/A')}\n"
            f"Web Title: {section.get('webTitle', 'N/A')}\n"
            f"Web URL: {section.get('webUrl', 'N/A')}\n"
            "--------------------------------------------------\n"
        )
        output_lines.append(entry)
        logging.info(f"{i+1}. {section.get('webTitle', 'N/A')} | Section: {section.get('sectionName', 'N/A')}")
    return output_lines


def main() -> None:
    """
    Main execution for fetching and saving ArXiv abstracts and The Guardian sections.
    API keys should be set via environment variables for security.
    """
    arxiv_entries = fetch_arxiv_abstracts(num_results=10, query="RAG AND LLM")
    if arxiv_entries:
        save_entries_to_file(arxiv_entries, filename="arxiv_abstracts.txt")
    else:
        logging.warning("No abstracts to save.")

    api_key = os.environ.get("GUARDIAN_API_KEY")
    guardian_entries = fetch_theguardian_sections(api_key=api_key)
    if guardian_entries:
        save_entries_to_file(guardian_entries, filename="theguardian_sections.txt")
    else:
        logging.warning("No sections to save.")

if __name__ == "__main__":
    main()