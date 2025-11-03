import datetime
from models import RawItems
from typing import List, Dict, Any
from utils import filter_distinct_items
from preprocessor import chunk_text, clean_html
from connectors import ArxivFetcher, GuardianFetcher


def run_arxiv_ingestion(keywords: List[str], db_conn) -> None:
    """
    Ingest articles from Arxiv based on keywords and store distinct items in the database.

    Args:
        keywords (List[str]): List of keywords to search for.
        db_conn: Database connection object.
    """
    fetcher = ArxivFetcher()
    raw_items_model = RawItems()

    for keyword in keywords:
        articles = fetcher.fetch_abstracts_metadata(num_results=1, query=keyword)
        # Use 'title' as the unique field for filtering
        distinct_articles = filter_distinct_items(db_conn, "raw_items", "title", articles)

        for article in distinct_articles:
            cleaned_abstract = clean_html(article['abstract'])
            chunks = chunk_text(cleaned_abstract)

            for chunk in chunks:
                query = """
                    INSERT INTO raw_items (source, source_id, fetched_at, title, authors, published_at, summary, full_text, url, processed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    'arxiv',
                    article.get('entry_id', ''),
                    article.get('fetched_time', ''),
                    article.get('title', ''),
                    ', '.join(article.get('authors', [])),
                    article.get('published', None),
                    chunk,
                    None,
                    article.get('pdf_url', ''),
                    False
                )
                raw_items_model.execute_non_query(query, params)

    raw_items_model.close()
    
    
if __name__ == "__main__":
    run_arxiv_ingestion(["LLM AND AI"], RawItems().connection)