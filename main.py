import arxiv
import logging
from datetime import datetime
from typing import List


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
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(entries)
        logging.info(f"Successfully stored all data in {filename}")
    except IOError as e:
        logging.error(f"Error writing to file: {e}")


def main(num_results: int = 10, query: str = "RAG AND LLM", filename: str = "arxiv_abstracts.txt") -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    entries = fetch_arxiv_abstracts(num_results=num_results, query=query)
    if entries:
        save_abstracts_to_file(entries, filename)
    else:
        logging.warning("No abstracts to save.")


if __name__ == "__main__":
    main(num_results=10, query="RAG AND LLM")