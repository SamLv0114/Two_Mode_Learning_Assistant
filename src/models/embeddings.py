"""
Embedding utilities for vector search
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from src.utils.config import settings
from src.utils.preprocessing import clean_text, chunk_text
import logging
import threading
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _scope_where(filter_type: Optional[str], user_id: Optional[int]) -> Optional[Dict]:
    """
    Build a ChromaDB `where` clause that scopes "user_doc" chunks to their
    owner. "paper"/"article" are global, shared content and are never scoped.

    A caller that knows filter_type == "user_doc" but omits user_id gets a
    clause matching no real document, rather than silently falling back to
    an unscoped query — the missing identity is a caller bug, and this
    fails closed instead of leaking every user's uploads.
    """
    if filter_type == "user_doc":
        return {"$and": [{"type": "user_doc"}, {"user_id": user_id if user_id is not None else -1}]}
    if filter_type:
        return {"type": filter_type}
    if user_id is not None:
        # Unfiltered search still must not leak other users' uploads: any
        # non-user_doc hit passes through, a user_doc hit only if it's this
        # user's own.
        return {
            "$or": [
                {"type": {"$ne": "user_doc"}},
                {"$and": [{"type": "user_doc"}, {"user_id": user_id}]},
            ]
        }
    # No identity at all (trusted system/background path) — exclude user_doc
    # entirely rather than search across every user's private uploads.
    return {"type": {"$ne": "user_doc"}}


class EmbeddingManager:
    """Manages embeddings and vector database
    
    Thread-safe singleton pattern: model is loaded once and shared across instances.
    """
    
    # Class-level shared model instance and lock
    _model_instance = None
    _model_lock = threading.Lock()
    _client_instance = None
    _client_lock = threading.Lock()
    _cache_lock = threading.Lock()  # Lock for cache operations
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        # Use shared model instance (thread-safe loading)
        self.model = self._get_or_load_model()
        
        # Initialize ChromaDB (shared client to avoid multiple connections)
        self.client = self._get_or_create_client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_DB_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    @classmethod
    def _get_or_load_model(cls) -> SentenceTransformer:
        """Thread-safe model loading - ensures only one instance loads at a time"""
        # Double-checked locking pattern
        if cls._model_instance is not None:
            return cls._model_instance
        
        with cls._model_lock:
            # Check again inside lock (another thread might have loaded it)
            if cls._model_instance is not None:
                return cls._model_instance
            
            logger.info(f"Loading model: {settings.EMBEDDING_MODEL} (first initialization)")
            cls._model_instance = cls._load_model_with_retry_impl(settings.EMBEDDING_MODEL)
            return cls._model_instance
    
    @classmethod
    def _get_or_create_client(cls):
        """Thread-safe ChromaDB client creation"""
        if cls._client_instance is not None:
            return cls._client_instance
        
        with cls._client_lock:
            if cls._client_instance is not None:
                return cls._client_instance
            
            logger.info("Initializing ChromaDB client (first initialization)")
            cls._client_instance = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            return cls._client_instance
    
    @staticmethod
    def _load_model_with_retry_impl(model_name: str) -> SentenceTransformer:
        """Internal implementation of model loading (called within lock)"""
        # 1) Try local cache first (fast path)
        try:
            model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
            logger.info("Model loaded from local cache")
            return model
        except Exception as e:
            msg = str(e).lower()
            logger.warning(f"Local cache load failed: {e}")

            # If it's the meta tensor / partial-cache kind of error, clear model cache
            # Use lock to prevent multiple instances from clearing cache simultaneously
            if "meta tensor" in msg or "cannot copy out of meta tensor" in msg:
                logger.warning("Meta-tensor / corrupted cache detected. Clearing model snapshot cache...")
                cache_path = (
                    Path.home() / ".cache" / "huggingface" / "hub" /
                    f"models--{model_name.replace('/', '--')}"
                )
                # Lock cache clearing to prevent race conditions
                with EmbeddingManager._cache_lock:
                    if cache_path.exists():
                        shutil.rmtree(cache_path, ignore_errors=True)

            # 2) Re-download clean copy (slow path)
            logger.info("Downloading model from Hugging Face...")
            model = SentenceTransformer(model_name, device="cpu")  # allow download
            logger.info("Model loaded successfully after download")
            return model


    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.model.encode(text, show_progress_bar=False).tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def add_paper(self, paper_id: str, title: str, abstract: str, metadata: Dict):
        """Add a paper to the vector database (updates if exists)"""
        paper_id_str = f"paper_{paper_id}"
        
        # Check if already exists
        try:
            existing = self.collection.get(ids=[paper_id_str])
            if existing["ids"]:
                # Update existing
                text = f"{title}\n\n{abstract}"
                embedding = self.generate_embedding(text)
                self.collection.update(
                    ids=[paper_id_str],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{
                        **metadata,
                        "type": "paper",
                        "paper_id": paper_id
                    }]
                )
                return
        except Exception:
            pass
        
        # Add new
        text = f"{title}\n\n{abstract}"
        embedding = self.generate_embedding(text)
        
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    **metadata,
                    "type": "paper",
                    "paper_id": paper_id
                }],
                ids=[paper_id_str]
            )
        except Exception as e:
            logger.debug(f"Error adding paper {paper_id}: {e}")
    
    def add_article(self, article_id: str, title: str, content: str, metadata: Dict):
        """Add an article to the vector database (updates if exists)"""
        article_id_str = f"article_{article_id}"
        
        # Check if already exists
        try:
            existing = self.collection.get(ids=[article_id_str])
            if existing["ids"]:
                # Update existing
                text = f"{title}\n\n{content[:1000]}"
                embedding = self.generate_embedding(text)
                self.collection.update(
                    ids=[article_id_str],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{
                        **metadata,
                        "type": "article",
                        "article_id": article_id
                    }]
                )
                return
        except Exception:
            pass
        
        # Add new
        text = f"{title}\n\n{content[:1000]}"
        embedding = self.generate_embedding(text)
        
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    **metadata,
                    "type": "article",
                    "article_id": article_id
                }],
                ids=[article_id_str]
            )
        except Exception as e:
            logger.debug(f"Error adding article {article_id}: {e}")

    def add_user_document(self, doc_id: str, title: str, content: str, metadata: Dict) -> int:
        """
        Add a user-uploaded document to the vector database as chunked entries.

        Returns:
            Number of chunks added
        """
        clean_content = clean_text(content)
        if not clean_content:
            return 0

        chunks = chunk_text(
            clean_content,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        if not chunks:
            return 0

        embeddings = self.generate_embeddings(chunks)
        ids = [f"userdoc_{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{
            **metadata,
            "type": "user_doc",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": i,
        } for i in range(len(chunks))]

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.debug(f"Error adding user document {doc_id}: {e}")
            return 0

        return len(chunks)

    def has_paper_fulltext(self, arxiv_id: str) -> bool:
        """Whether this paper's full text has already been fetched, parsed, and cached."""
        try:
            existing = self.collection.get(
                where={"$and": [{"type": "paper_fulltext"}, {"arxiv_id": arxiv_id}]},
                limit=1,
                include=[],
            )
            return bool(existing.get("ids"))
        except Exception as e:
            logger.debug(f"has_paper_fulltext check failed for {arxiv_id}: {e}")
            return False

    def add_paper_fulltext(self, arxiv_id: str, title: str, full_text: str, metadata: Dict) -> int:
        """
        Chunk and embed a paper's full body text (beyond the title+abstract
        add_paper stores), for on-demand deep lookups via the fetch_full_paper
        tool. Stored under a distinct "paper_fulltext" type + arxiv_id filter
        so it never surfaces in default search_knowledge_base calls — those
        callers expect a handful of whole-paper hits, not dozens of chunks
        from one paper's body.

        Returns:
            Number of chunks added (0 if the paper has no extractable text).
        """
        clean_content = clean_text(full_text)
        if not clean_content:
            return 0

        chunks = chunk_text(
            clean_content,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        if not chunks:
            return 0

        embeddings = self.generate_embeddings(chunks)
        ids = [f"paperfull_{arxiv_id}_{i}" for i in range(len(chunks))]
        metadatas = [{
            **metadata,
            "type": "paper_fulltext",
            "arxiv_id": arxiv_id,
            "title": title,
            "chunk_index": i,
        } for i in range(len(chunks))]

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            logger.debug(f"Error adding fulltext for {arxiv_id}: {e}")
            return 0

        return len(chunks)

    def search_paper_fulltext(self, arxiv_id: str, query: str, n_results: int = 5) -> List[Dict]:
        """Vector search restricted to one paper's cached full-text chunks."""
        query_embedding = self.generate_embedding(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"$and": [{"type": "paper_fulltext"}, {"arxiv_id": arxiv_id}]},
            )
        except Exception as e:
            logger.warning(f"search_paper_fulltext failed for {arxiv_id}: {e}")
            return []

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })
        return formatted_results

    def delete_user_document(self, doc_id: str) -> int:
        """
        Delete all ChromaDB chunks that belong to a user document.

        Args:
            doc_id: The doc_id value stored in chunk metadata (e.g. "userdoc_3_a1b2c3d4").

        Returns:
            Number of chunks deleted (0 if none found or on error).
        """
        try:
            results = self.collection.get(where={"doc_id": doc_id}, include=[])
            chunk_ids = results.get("ids", [])
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks for doc_id={doc_id}")
            return len(chunk_ids)
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB chunks for doc_id={doc_id}: {e}")
            return 0

    def search(
        self, query: str, n_results: int = 10, filter_type: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search the vector database

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_type: Optional filter by "paper", "article", or "user_doc"
            user_id: Requesting user's id. "user_doc" chunks are private per-user
                (unlike "paper"/"article", which are global) — this scopes any
                user_doc results to this user so one user's uploads never surface
                in another user's search results. Pass the current user's id on
                every user-facing call; omit only for trusted system/background
                paths that intentionally have no per-user scope.

        Returns:
            List of search results with metadata
        """
        query_embedding = self.generate_embedding(query)

        where = _scope_where(filter_type, user_id)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
    
    def get_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.model.encode(text1, show_progress_bar=False)
        emb2 = self.model.encode(text2, show_progress_bar=False)
        
        # Cosine similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

