import faiss
import logging
import numpy as np
from uuid import uuid4
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Protocol, Sequence, Iterable, Union

from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import TextEmbedEmbeddings as LCTextEmbedEmbeddings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define a protocol for the embeddings client to ensure compatibility
class EmbeddingsClient(Protocol):
    """Protocol describing the minimal embeddings interface we rely on."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - interface only
        ...

    def embed_query(self, text: str) -> Sequence[float]:  # pragma: no cover - interface only
        ...


# Wrapper for SentenceTransformer embeddings
class SentenceTransformerEmbeddings:
    """
    A wrapper for the SentenceTransformer library to provide a consistent interface
    for embedding generation.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes and loads the SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # Fail fast with a clear error; avoid silent None state.
            logger.exception("Error loading SentenceTransformer model '%s'", model_name)
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}'") from e

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of strings to be embedded.

        Returns:
            List[List[float]]: Embeddings for each input text.
        """
        # The encode method can take a single sentence or a list of sentences
        # and returns a numpy array of embeddings.
        embeddings = self.model.encode(list(texts), convert_to_numpy=True)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        # Fallback in unlikely types
        return [list(map(float, v)) for v in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single query string.

        Args:
            text (str): A single string to be embedded.

        Returns:
            List[float]: The embedding vector for the query.
        """
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec))


# Wrapper for TextEmbedEmbeddings from langchain_community
class TextEmbedEmbeddingsClient:
    """
    A wrapper for the TextEmbedEmbeddings from langchain_community to provide a consistent interface
    for embedding generation via a TextEmbed server.
    """
    def __init__(self, model: str, api_url: str, api_key: str):
        """
        Initializes the TextEmbedEmbeddings client.

        Args:
            model (str): The model name to use on the TextEmbed server.
            api_url (str): The base URL of the TextEmbed server.
            api_key (str): The API key for authentication.
        """
        self.embeddings = LCTextEmbedEmbeddings(
            model=model,
            api_url=api_url,
            api_key=api_key,
        )
        
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): A list of strings to be embedded.
        """
        return list(self.embeddings.embed_documents(list(texts)))
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single query string.

        Args:
            text (str): A single string to be embedded.
        """
        return list(self.embeddings.embed_query(text))
    

# Wrapper for FAISS vector store
class FaissVectorStore:
    """
    A wrapper for FAISS vector store to manage document embeddings.
    """
    def __init__(
        self,
        embeddings: EmbeddingsClient,
        *,
        index: Optional[faiss.Index] = None,    
        dimension: Optional[int] = None,
    ):
        """
        Initializes the FAISS vector store.

        Args:
            embeddings: The embedding function/client to use for generating embeddings.
            index: Optional pre-constructed FAISS index.
            dimension: Optional embedding dimensionality to initialize a flat index
                when no index is provided.
        """
        self._embeddings = embeddings

        if index is not None:
            faiss_index = index
        else:
            try:
                dim = (
                    int(dimension)
                    if dimension is not None
                    else len(embeddings.embed_query("hello world")) # Infer dimension
                )
            except Exception as e:
                logger.exception("Failed to infer embedding dimension from client")
                raise RuntimeError(
                    "Unable to determine embedding dimension. Provide 'dimension' or a valid 'index'."
                ) from e
            faiss_index = faiss.IndexFlatL2(dim)

        # LangChain FAISS expects a callable for embedding_function in some versions.
        # Provide a callable that delegates to the client's embed_query.
        def _embed_fn(text: str) -> List[float]:
            return list(embeddings.embed_query(text))

        self.vector_store = FAISS(
            embedding_function=_embed_fn,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_embeddings(self, documents: List[Document], ids: Optional[Sequence[str]] = None) -> None:
        """
        Adds documents to the vector store.

        Args:
            documents (List[Document]): A list of Document objects to add.
            ids (Optional[List[str]]): Optional list of IDs for the documents.
        """
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))] # Generate unique IDs if not provided
        
        metadatas = [doc.metadata for doc in documents]  
        texts = [doc.page_content for doc in documents]
        
        vectors = self._embeddings.embed_documents(texts)
        text_embeddings = list(zip(texts, vectors))

        self.vector_store.add_embeddings(text_embeddings=text_embeddings, ids=list(ids)) 
        
        return dict(zip(ids, vectors))

    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[Document]:
        """
        Performs a similarity search in the vector store.

        Args:
            query (str): The query string to search for.
            k (int): The number of top similar documents to retrieve.
            filter (Optional[dict]): An optional filter to apply to the search.
        
        Returns:
            List[Document]: A list of the top k similar Document objects.
        """
        return self.vector_store.similarity_search(query, k, filter=filter)
    
    def save_local(self, folder_path: Union[str, Path]) -> None:
        """
        Saves the vector store locally to the specified folder.

        Args:
            folder_path (str | Path): Path to the folder where the vector store will be saved.
        """
        self.vector_store.save_local(str(folder_path))
    
    @classmethod
    def from_local(
        cls,
        folder_path: Union[str, Path],
        embeddings: EmbeddingsClient,
        allow_dangerous_deserialization: bool = False,
    ) -> "FaissVectorStore":
        """
        Construct a FaissVectorStore by loading an existing FAISS index from disk.

        Args:
            folder_path (str | Path): Directory containing the saved FAISS index and metadata.
            embeddings: Embeddings client compatible with LangChain's FAISS.load_local.
            allow_dangerous_deserialization (bool): Required by newer LangChain versions to
                enable pickle deserialization when loading. Defaults to False.

        Returns:
            FaissVectorStore: A wrapper instance around the loaded FAISS store.
        """
        path = Path(folder_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS folder not found: {folder_path}")

        loaded = FAISS.load_local(
            str(path),
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )

        instance = cls.__new__(cls)
        instance._embeddings = embeddings
        instance.vector_store = loaded
        
        try:
            instance.vector_store.embedding_function = lambda text: list(embeddings.embed_query(text))
        except Exception:
            pass
        return instance

    def load_local(self, folder_path: Union[str, Path], allow_dangerous_deserialization: bool = False) -> None:
        """
        Load a FAISS index from disk into this instance, replacing the current store.

        Args:
            folder_path (str | Path): Directory containing the saved FAISS index and metadata.
            allow_dangerous_deserialization (bool): Required by newer LangChain versions to
                enable pickle deserialization when loading. Defaults to False.
        """
        path = Path(folder_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS folder not found: {folder_path}")

        self.vector_store = FAISS.load_local(
            str(path),
            embeddings=self._embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )
        
        try:
            self.vector_store.embedding_function = lambda text: list(self._embeddings.embed_query(text))
        except Exception:
            pass