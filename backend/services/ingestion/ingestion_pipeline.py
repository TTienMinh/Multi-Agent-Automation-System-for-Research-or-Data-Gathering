import datetime
from typing import List, Dict, Any

from backend.services.ingestion.models import RawItems
from backend.services.ingestion.utils import filter_distinct_items
from backend.services.ingestion.preprocessor import chunk_text, clean_html
from backend.services.ingestion.connectors import ArxivFetcher, PubMedFetcher, GuardianFetcher

from backend.services.embedding.models import FaissVectorStore, TextEmbedEmbeddingsClient


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

        # Insert distinct articles into the database
        metadata_params_list = []
        for article in distinct_articles:
            cleaned_abstract = clean_html(article['abstract'])

            insert_metadata_query = """
                INSERT INTO raw_items 
                (source, source_id, fetched_at, title, authors, published_at, summary, full_text, url, processed)
                VALUES %s
                RETURNING id
            """
            metadata_params_list.append((
                article.get('source', ''),
                article.get('id', ''),
                article.get('fetched_time', ''),
                article.get('title', ''),
                article.get('authors', ''),
                article.get('publication_date', None),
                cleaned_abstract,
                None,   # full_text
                article.get('url', ''),
                False   # processed
            ))
            articles_ids = raw_items_model.execute_bulk_insert(insert_metadata_query, metadata_params_list)
            
            print(f"Inserted {len(articles_ids)} articles. Generating chunks...")
            
            # Vector store and embeddings client initialization
            vector_store = FaissVectorStore()
            embeddings_client = TextEmbedEmbeddingsClient()
            
            # Generate and insert chunks for each article
            chunks_params_list = []
            for article, article_id in zip(distinct_articles, articles_ids):
                chunked_abstracts = chunk_text(cleaned_abstract, chunk_size=500, overlap=100)
                
                for idx, chunk in enumerate(chunked_abstracts):
                    chunks_params_list.append((
                        article_id,     # foreign key to raw_items.id
                        idx,            # chunk index
                        chunk,          # chunk text
                        None            # vector_id
                    ))
            
            insert_chunks_query = """
                INSERT INTO chunks (document_id, chunk_index, text, vector_id)
                VALUES %s
            """

            raw_items_model.execute_bulk_insert(insert_chunks_query, chunks_params_list)
            print(f"Inserted {len(chunks_params_list)} chunks successfully.")

    raw_items_model.close()
    
    
if __name__ == "__main__":
    run_arxiv_ingestion(["AI"], RawItems().connection)