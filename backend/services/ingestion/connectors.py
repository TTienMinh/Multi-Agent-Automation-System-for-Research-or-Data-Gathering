import os
import arxiv
import logging
from typing import List
from pathlib import Path
from datetime import datetime
from theguardian import theguardian_content, theguardian_section

project_root = Path(__file__).parent.parent.parent.parent
print(f"Project root directory: {project_root}")

def fetch_arxiv_abstracts(num_results: int = 10, query: str = "LLM") -> List[str]:
    """
    Fetches abstracts from ArXiv using the arxiv library.
    Returns a list of formatted paper entries.
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


def save_abstracts_to_file(entries: List[str], filename: str = "arxiv_abstracts.txt") -> None:
    """
    Saves the list of paper entries to a file.
    """
    full_path = project_root / "data" / filename
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(entries)
        logging.info(f"Successfully stored all data in {full_path}")
    except IOError as e:
        logging.error(f"Error writing to file: {e}")
        

def fetch_theguardian_sections() -> None:
    """
    Fetches and prints sections from The Guardian via API.
    """
    headers = {
        "q": "AI OR artificial intelligence",
        "tag": "technology/technology",
        "from-date": "2025-05-01",
        "order-by": "relevance",
    }
    content = theguardian_content.Content(api="test", **headers)
    res = content.get_content_response()
    result = content.get_results(res)

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

    for line in output_lines:
        print(line)
        print("--------------------------------------------------")
        
    return output_lines

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    entries = fetch_arxiv_abstracts(num_results=10, query="RAG AND LLM")
    if entries:
        save_abstracts_to_file(entries, filename="arxiv_abstracts.txt")
    else:
        logging.warning("No abstracts to save.")
        
    entries = fetch_theguardian_sections()
    if entries:
        save_abstracts_to_file(entries, filename="theguardian_sections.txt")
    else:
        logging.warning("No sections to save.")

if __name__ == "__main__":
    main()