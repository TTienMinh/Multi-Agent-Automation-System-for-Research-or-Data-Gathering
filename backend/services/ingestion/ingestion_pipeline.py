import os
import dotenv
import logging
import datetime
from typing import List, Dict, Tuple, Any
from langchain_core.documents import Document

from backend.services.ingestion.models import RawItems
from backend.services.ingestion.utils import filter_distinct_items
from backend.services.ingestion.preprocessor import chunk_text, clean_html
from backend.services.ingestion.connectors import ArxivFetcher, PubMedFetcher, GuardianFetcher

from backend.services.embedding.models import FaissVectorStore, TextEmbedEmbeddingsClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_and_merge_articles(keywords: List[str], arxiv_num_results: int, pubmed_num_results: int) -> List[Dict]:
    """
    Responsibilities: API interaction and merging results.
    """
    arxiv_fetcher = ArxivFetcher()
    pubmed_fetcher = PubMedFetcher()
    all_articles = []

    logger.info(f"Starting fetch for {len(keywords)} keywords.")
    
    for keyword in keywords:
        try:
            # Fetch from both sources
            arxiv_results = arxiv_fetcher.fetch_articles_metadata(num_results=arxiv_num_results, query=keyword)
            pubmed_results = pubmed_fetcher.fetch_articles_metadata(max_results=pubmed_num_results, query=keyword)

            count = len(arxiv_results) + len(pubmed_results)
            logger.debug(f"Fetched {count} articles for keyword '{keyword}'.")
            
            all_articles.extend(arxiv_results + pubmed_results)
        except Exception as e:
            logger.error(f"Error fetching articles for keyword '{keyword}': {e}", exc_info=True)

    return all_articles


def save_articles_metadata(db_model, articles: List[Dict]) -> List[int]:
    """
    Responsibilities: Database operations for saving metadata.
    """
    if not articles:
        logger.info("No new articles to save.")
        return []
    
    metadata_params_list = []
    for article in articles:
        cleaned_abstract = clean_html(article['abstract'])
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
        
    insert_metadata_query = """
        INSERT INTO raw_items 
        (source, source_id, fetched_at, title, authors, published_at, summary, full_text, url, processed)
        VALUES %s
        RETURNING id
    """
    
    try:
        ids = db_model.execute_bulk_insert(insert_metadata_query, metadata_params_list)
        logger.info(f"Inserted {len(ids)} new articles into the database.")
        return ids
    except Exception as e:
        logger.critical(f"Failed to insert articles metadata: {e}")
        raise
    

def process_and_store_chunks(db_model, articles: List[Dict], article_ids: List[int], vector_store: FaissVectorStore) -> None:
    """
    Responsibilities: Text chunking, embedding generation, and storing chunks.
    """
    if not articles or not article_ids:
        logger.warning("No articles or article IDs provided for chunk processing.")
        return
    
    chunks_params_list = []
    logger.info(f"Processing chunks for {len(articles)} articles.")
    
    for article, article_id in zip(articles, article_ids):
        abstract = article.get('cleaned_abstract', '')
        if not abstract:
            continue
            
        chunked_text = chunk_text(abstract, chunk_size=500, overlap=100)

        for idx, chunk in enumerate(chunked_text):
            if not chunk.strip():
                continue
            try:
                # Embed the chunk
                embedded_vector = vector_store.add_embeddings([Document(page_content=chunk)])
                chunks_params_list.append((
                    article_id,                         # foreign key to raw_items.id
                    idx,                                # chunk index
                    chunk,                              # chunk text
                    list(embedded_vector.keys())[0]     # vector_id
                ))
            except Exception as e:
                logger.error(f"Failed to embed chunk {idx} for article ID {article_id}: {e}")
                
    if chunks_params_list:
        insert_chunks_query = """
            INSERT INTO chunks (document_id, chunk_index, text, vector_id)
            VALUES %s
        """
        
        try:
            db_model.execute_bulk_insert(insert_chunks_query, chunks_params_list)
            logger.info(f"Inserted {len(chunks_params_list)} chunks for article ID {article_id}.")
        except Exception as e:
            logger.critical(f"Failed to insert chunks for article ID {article_id}: {e}")
            raise
            

def run_ingestion_pipeline(
        keywords: List[str], 
        db_conn, 
        api_url, 
        api_key,
        arxiv_num_results: int = 2, 
        pubmed_num_results: int = 2
    ) -> None:
    """
    Main entry point. Coordinates the flow of data.
    """
    logger.info("Starting ingestion job.")
    
    # Initialization
    raw_items_model = RawItems()
    embeddings_client = TextEmbedEmbeddingsClient(
        model="sentence-transformers/all-MiniLM-L6-v2",
        api_url=api_url,
        api_key=api_key
    )
    vector_store = FaissVectorStore(embeddings=embeddings_client)
    
    try:
        # Step 1: Fetch and merge articles
        all_articles = fetch_and_merge_articles(keywords, arxiv_num_results, pubmed_num_results)
        
        # Step 2: Filter distinct articles
        unique_articles = filter_distinct_items(db_conn, "raw_items", "title", all_articles)
        
        if not unique_articles:
            logger.info("Job finished: No new unique articles found.")
            return

        logger.info(f"Processing {len(unique_articles)} articles.")
        
        # Step 3: Prep
        for article in unique_articles:
            article['cleaned_abstract'] = clean_html(article['abstract'])
        
        # Step 4: Save metadata
        article_ids = save_articles_metadata(raw_items_model, unique_articles)
        
        # Step 5: Save Chunks
        process_and_store_chunks(raw_items_model, unique_articles, article_ids, vector_store)
        
        logger.info("Ingestion job completed successfully.")

    except Exception as e:
        logger.critical(f"Critical failure in ingestion job: {e}", exc_info=True)
    
    finally:
        raw_items_model.close()
        logger.info("Database connection closed.")
        
    
if __name__ == "__main__":
    run_ingestion_pipeline(
        ["AI"], 
        RawItems().connection, 
        os.getenv("TEXT_EMBED_API_URL"), 
        os.getenv("TEXT_EMBED_API_KEY"),
        arxiv_num_results=2, 
        pubmed_num_results=2
    )