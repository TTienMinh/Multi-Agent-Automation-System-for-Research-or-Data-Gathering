import os
import time
import arxiv
import dotenv
import logging
from typing import List
from pathlib import Path
from datetime import datetime
from pymed_paperscraper import PubMed
from theguardian import theguardian_content, theguardian_section

project_root = Path(__file__).parent.parent.parent.parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArxivFetcher:
    """
    Class to fetch articles from ArXiv and save them to file.
    """
    def __init__(self):
        self.client = arxiv.Client()

    def fetch_articles_metadata(self, num_results: int = 10, query: str = "LLM") -> List[dict]:
        """
        Fetch articles from ArXiv and return a list of metadata dicts.

        Args:
            num_results (int): Number of results to fetch.
            query (str): Search query for ArXiv.
        Returns:
            List[dict]: List of metadata dicts for each paper.
        """
        search = arxiv.Search(
            query=query,
            max_results=num_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        try:
            all_results = list(self.client.results(search))
        except Exception as e:
            logging.error(f"Error fetching results from arXiv: {e}")
            return []

        if not all_results:
            logging.warning("No results found.")
            return []

        metadata_list = []
        for paper in all_results:
            metadata = {
                "source": "arxiv",
                "title": paper.title,
                "url": paper.entry_id,
                "id": paper.entry_id.split('/')[-1],
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary.replace('\n', ' '),
                "publication_date": paper.published.strftime('%Y-%m-%d'),
                "fetched_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            metadata_list.append(metadata)
        return metadata_list

    
    def save_metadata_to_file(self, metadata_list: List[dict], filename: str) -> None:
        """
        Save a list of metadata dicts to a file as formatted text.

        Args:
            metadata_list (List[dict]): List of metadata dicts.
            filename (str): Name of the file to save entries to.
        """
        full_path = project_root / "data" / filename
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                for i, meta in enumerate(metadata_list):
                    entry = (
                        f"Paper {i+1}/{len(metadata_list)}:\n"
                        f"Title: {meta.get('title', '')}\n"
                        f"Authors: {', '.join(meta.get('authors', []))}\n"
                        f"ID: {meta.get('entry_id', '')}\n"
                        f"Published: {meta.get('published', '')}\n"
                        f"Fetched Time: {meta.get('fetched_time', '')}\n"
                        f"PDF URL: {meta.get('pdf_url', '')}\n"
                        f"Abstract:\n{meta.get('abstract', '')}\n"
                        "--------------------------------------------------\n"
                    )
                    f.write(entry)
            logging.info(f"Successfully stored all data in {full_path}")
        except IOError as e:
            logging.error(f"Error writing to file: {e}")


class PubMedFetcher:
    """
    Class to fetch articles from PubMed.
    """
    def __init__(self):
        self.client = PubMed()
        
    def fetch_articles_metadata(self, query: str = "AI", max_results: int = 10) -> List[dict]:
        search = list(self.client.query(query, max_results=max_results))
        metadata_list = []
        for article in search:
            authors = []
            for author in getattr(article, "authors", []):
                authors.append(" ".join([author.get('firstname', ''), author.get('lastname', '')]).strip())
            metadata = {
                "source": "pubmed",
                "title": getattr(article, "title", ""),
                "authors": authors,
                "abstract": getattr(article, "abstract", ""),
                "id": getattr(article, "pubmed_id", ""),
                "publication_date": getattr(article, "publication_date", ""),
                "fetched_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "url": "https://doi.org/" + getattr(article, "doi", "") if getattr(article, "doi", "") else "",
            }
            metadata_list.append(metadata)
        return metadata_list
    
    
    def save_metadata_to_file(self, metadata_list: List[dict], filename: str) -> None:
        full_path = project_root / "data" / filename
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                for i, meta in enumerate(metadata_list):
                    entry = (
                        f"Article {i+1}/{len(metadata_list)}:\n"
                        f"Title: {meta.get('title', '')}\n"
                        f"Authors: {', '.join(meta.get('authors', []))}\n"
                        f"PubMed ID: {meta.get('pubmed_id', '')}\n"
                        f"Publication Date: {meta.get('publication_date', '')}\n"
                        f"Fetched Time: {meta.get('fetched_time', '')}\n"
                        f"URL: {meta.get('url', '')}\n"
                        f"Abstract:\n{meta.get('abstract', '')}\n"
                        "--------------------------------------------------\n"
                    )
                    f.write(entry)
            logging.info(f"Successfully stored all data in {full_path}")
        except IOError as e:
            logging.error(f"Error writing to file: {e}")


class GuardianFetcher:
    """
    Class to fetch sections from The Guardian and save them to file.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch_sections_metadata(
        self,
        query: str = "AI OR artificial intelligence",
        tag: str = "technology/technology",
        from_date: str = "2025-05-01",
        order_by: str = "relevance"
    ) -> List[dict]:
        """
        Fetch sections from The Guardian and return a list of metadata dicts.

        Args:
            api_key (str): The Guardian API key.
            query (str): Search query.
            tag (str): Tag to filter content.
            from_date (str): Start date for articles (YYYY-MM-DD). Defaults to today if None.
            order_by (str): Order of results.
        Returns:
            List[dict]: List of metadata dicts for each section.
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
            content = theguardian_content.Content(api=self.api_key, **headers)
            res = content.get_content_response()
            result = content.get_results(res)
        except Exception as e:
            logging.error(f"Error fetching results from The Guardian: {e}")
            return []

        metadata_list = []
        for section in result:
            metadata = {
                "source": "theguardian",
                "id": section.get('id', 'N/A'),
                "sectionId": section.get('sectionId', 'N/A'),
                "sectionName": section.get('sectionName', 'N/A'),
                "webPublicationDate": section.get('webPublicationDate', 'N/A'),
                "webTitle": section.get('webTitle', 'N/A'),
                "webUrl": section.get('webUrl', 'N/A'),
                "fetched_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            metadata_list.append(metadata)
        return metadata_list

    
    def save_metadata_to_file(self, metadata_list: List[dict], filename: str) -> None:
        """
        Save a list of metadata dicts to a file as formatted text.

        Args:
            metadata_list (List[dict]): List of metadata dicts.
            filename (str): Name of the file to save entries to.
        """
        full_path = project_root / "data" / filename
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                for i, meta in enumerate(metadata_list):
                    entry = (
                        f"Paper No. {i+1}/{len(metadata_list)}:\n"
                        f"ID: {meta.get('id', 'N/A')}\n"
                        f"Section ID: {meta.get('sectionId', 'N/A')}\n"
                        f"Section Name: {meta.get('sectionName', 'N/A')}\n"
                        f"Publication Date: {meta.get('webPublicationDate', 'N/A')}\n"
                        f"Web Title: {meta.get('webTitle', 'N/A')}\n"
                        f"Web URL: {meta.get('webUrl', 'N/A')}\n"
                        f"Fetched Time: {meta.get('fetched_time', 'N/A')}\n"
                        "--------------------------------------------------\n"
                    )
                    f.write(entry)
            logging.info(f"Successfully stored all data in {full_path}")
        except IOError as e:
            logging.error(f"Error writing to file: {e}")


def main() -> None:
    """
    Main execution for fetching and saving ArXiv abstracts and The Guardian sections.
    API keys should be set via environment variables for security.
    """
    arxiv_metadata = ArxivFetcher.fetch_abstracts_metadata(num_results=10, query="RAG AND LLM")
    if arxiv_metadata:
        ArxivFetcher.save_metadata_to_file(arxiv_metadata, filename="arxiv_abstracts.txt")
    else:
        logging.warning("No abstracts to save.")

    api_key = os.environ.get("GUARDIAN_API_KEY")
    guardian_metadata = GuardianFetcher.fetch_sections_metadata(api_key=api_key)
    if guardian_metadata:
        GuardianFetcher.save_metadata_to_file(guardian_metadata, filename="theguardian_sections.txt")
    else:
        logging.warning("No sections to save.")

if __name__ == "__main__":
    main()