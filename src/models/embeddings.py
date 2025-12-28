"""
Embedding utilities for vector search
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from src.utils.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector database"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_DB_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
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
    
    def search(self, query: str, n_results: int = 10, filter_type: Optional[str] = None) -> List[Dict]:
        """
        Search the vector database
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_type: Optional filter by "paper" or "article"
            
        Returns:
            List of search results with metadata
        """
        query_embedding = self.generate_embedding(query)
        
        where = None
        if filter_type:
            where = {"type": filter_type}
        
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

