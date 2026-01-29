"""
Configuration settings for the Multi-Document RAG System.

This module provides centralized configuration management using Pydantic Settings,
loading values from environment variables with sensible defaults.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All configuration values can be overridden via environment variables
    or a .env file in the project root.
    """
    
    # LLM Configuration
    groq_api_key: str = Field(
        default="",
        description="API key for Groq LLM service"
    )
    llm_model: str = Field(
        default="llama-3.1-70b-versatile",
        description="LLM model name for answer generation"
    )
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=500,
        description="Maximum number of characters per chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Number of overlapping characters between chunks"
    )
    
    # Retrieval Configuration
    top_k_results: int = Field(
        default=5,
        description="Number of top similar chunks to retrieve"
    )
    
    # Vector Store Configuration
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="Directory path for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="documents",
        description="Name of the ChromaDB collection"
    )
    
    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        description="Server port number"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure settings are only loaded once
    and reused across the application.
    
    Returns:
        Settings: The application settings instance.
    """
    return Settings()
