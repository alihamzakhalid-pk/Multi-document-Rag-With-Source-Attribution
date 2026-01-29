"""
Vector Store for the Multi-Document RAG System.

Manages document storage and retrieval using ChromaDB for 
persistent vector storage and similarity search.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings
from utils.logger import get_logger
from .chunker import TextChunk
from .embeddings import EmbeddingGenerator

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for document chunks.
    
    Provides persistent storage for document embeddings with
    metadata, enabling efficient similarity search.
    """
    
    _instance = None
    _client = None
    _collection = None
    
    def __new__(cls):
        """Singleton pattern for consistent store access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store with ChromaDB."""
        if VectorStore._client is None:
            settings = get_settings()
            persist_dir = settings.chroma_persist_dir
            collection_name = settings.chroma_collection_name
            
            # Ensure directory exists
            os.makedirs(persist_dir, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at: {persist_dir}")
            
            # Initialize ChromaDB client with persistence
            VectorStore._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            VectorStore._collection = VectorStore._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(
                f"ChromaDB initialized. Collection '{collection_name}' has "
                f"{VectorStore._collection.count()} documents"
            )
        
        self.embedding_generator = EmbeddingGenerator()
    
    @property
    def collection(self):
        """Get the ChromaDB collection."""
        return VectorStore._collection
    
    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects to add.
        
        Returns:
            int: Number of chunks added.
        """
        if not chunks:
            return 0
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embedding_generator.generate(texts)
        
        # Add to collection
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks")
        return len(chunks)
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_doc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks to the query.
        
        Args:
            query: The search query text.
            top_k: Number of results to return. Defaults to settings value.
            filter_doc: Optional document name to filter results.
        
        Returns:
            List[Dict]: List of results with text, metadata, and distance.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single(query)
        
        # Build filter if specified
        where_filter = None
        if filter_doc:
            where_filter = {"document_name": filter_doc}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })
        
        logger.debug(f"Search returned {len(formatted_results)} results")
        return formatted_results
    
    def get_all_documents(self) -> List[str]:
        """
        Get a list of all unique document names in the store.
        
        Returns:
            List[str]: Unique document names.
        """
        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        
        if not all_data or not all_data.get("metadatas"):
            return []
        
        # Extract unique document names
        doc_names = set()
        for metadata in all_data["metadatas"]:
            if metadata and "document_name" in metadata:
                doc_names.add(metadata["document_name"])
        
        return sorted(list(doc_names))
    
    def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_name: Name of the document.
        
        Returns:
            List[Dict]: All chunks belonging to the document.
        """
        results = self.collection.get(
            where={"document_name": document_name},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                chunks.append({
                    "chunk_id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        
        return chunks
    
    def delete_document(self, document_name: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Args:
            document_name: Name of the document to delete.
        
        Returns:
            int: Number of chunks deleted.
        """
        # Get all chunks for the document
        chunks = self.get_document_chunks(document_name)
        
        if not chunks:
            logger.info(f"No chunks found for document: {document_name}")
            return 0
        
        # Delete by IDs
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.collection.delete(ids=chunk_ids)
        
        logger.info(f"Deleted {len(chunk_ids)} chunks for document: {document_name}")
        return len(chunk_ids)
    
    def count(self) -> int:
        """
        Get the total number of chunks in the store.
        
        Returns:
            int: Total chunk count.
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """Delete all documents from the collection."""
        settings = get_settings()
        collection_name = settings.chroma_collection_name
        
        # Delete and recreate collection
        VectorStore._client.delete_collection(collection_name)
        VectorStore._collection = VectorStore._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Vector store cleared")
