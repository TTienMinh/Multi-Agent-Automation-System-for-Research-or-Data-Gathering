import datetime
from models import RawItems
from typing import List, Dict, Any
from utils import filter_distinct_items
from preprocessor import chunk_text, clean_html
from connectors import ArxivFetcher, PubMedFetcher, GuardianFetcher


def run_arxiv_ingestion(keywords: List[str], db_conn) -> None:
    """
    Ingest articles from Arxiv based on keywords and store distinct items in the database.

    Args:
        keywords (List[str]): List of keywords to search for.
        db_conn: Database connection object.
    """
    arxiv_fetcher = ArxivFetcher()
    pubmed_fetcher = PubMedFetcher()
    raw_items_model = RawItems()

    for keyword in keywords:
        all_articles = arxiv_fetcher.fetch_articles_metadata(num_results=5, query=keyword) + \
                        pubmed_fetcher.fetch_articles_metadata(max_results=5, query=keyword)
        # Use 'title' as the unique field for filtering
        distinct_articles = filter_distinct_items(db_conn, "raw_items", "title", all_articles)

        for article in distinct_articles:
            cleaned_abstract = clean_html(article['abstract'])

            query = """
                INSERT INTO raw_items (source, source_id, fetched_at, title, authors, published_at, summary, full_text, url, processed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                article.get('source', ''),
                article.get('id', ''),
                article.get('fetched_time', ''),
                article.get('title', ''),
                ', '.join(article.get('authors', [])),
                article.get('publication_date', None),
                cleaned_abstract,
                None,
                article.get('url', ''),
                False
            )
            raw_items_model.execute_non_query(query, params)

    raw_items_model.close()
    
    
if __name__ == "__main__":
    run_arxiv_ingestion(["AI"], RawItems().connection)