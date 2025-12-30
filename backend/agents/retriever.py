from typing import List, Dict

from backend.services.embedding.models import FaissVectorStore

class RetrieverAgent:
    def __init__(self, vector_store: FaissVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from the vector store based on the query.
        """
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
