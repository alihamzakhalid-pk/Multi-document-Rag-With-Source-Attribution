"""
Embedding Generator for the Multi-Document RAG System.

Handles generation of vector embeddings for text chunks using
sentence-transformers models.
"""

from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings for text using sentence-transformers.
    
    Uses a configurable model to convert text into dense vector
    representations suitable for similarity search.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding generator with the configured model."""
        if EmbeddingGenerator._model is None:
            settings = get_settings()
            model_name = settings.embedding_model
            
            logger.info(f"Loading embedding model: {model_name}")
            EmbeddingGenerator._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the underlying sentence transformer model."""
        return EmbeddingGenerator._model
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if not texts:
            return []
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Convert to list format
        embeddings_list = embeddings.tolist()
        
        logger.debug(f"Generated {len(embeddings_list)} embeddings")
        return embeddings_list
    
    def generate_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
        
        Returns:
            List[float]: Embedding vector.
        """
        embeddings = self.generate([text])
        return embeddings[0] if embeddings else []
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: Number of dimensions in the embedding vector.
        """
        return self.model.get_sentence_embedding_dimension()
