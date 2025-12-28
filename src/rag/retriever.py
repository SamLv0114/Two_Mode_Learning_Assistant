"""
Vector search retriever for RAG
"""
from typing import List, Dict, Optional
from src.models.embeddings import EmbeddingManager
from src.utils.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant documents using vector search"""
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
    
    def retrieve(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_type: Optional filter by "paper" or "article"
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.embedding_manager.search(
            query=query,
            n_results=n_results,
            filter_type=filter_type
        )
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def retrieve_with_scores(self, query: str, n_results: int = 5, 
                            min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            n_results: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of documents filtered by score
        """
        results = self.retrieve(query, n_results=n_results)
        
        # Filter by minimum score (distance is inverse of similarity)
        filtered = []
        for result in results:
            # Convert distance to similarity (1 - distance for cosine)
            if result.get("distance") is not None:
                similarity = 1 - result["distance"]
                if similarity >= min_score:
                    result["similarity"] = similarity
                    filtered.append(result)
            else:
                filtered.append(result)
        
        return filtered

