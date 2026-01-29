"""
Retriever for the Multi-Document RAG System.

Handles retrieval of relevant document chunks for a given query,
formatting retrieved context for the answer generator.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from config.settings import get_settings
from utils.logger import get_logger
from .vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved chunk with full attribution.
    
    Attributes:
        document_name: Name of the source document.
        page_number: Page number from the original document.
        chunk_id: Unique identifier for this chunk.
        text: The chunk text content.
        score: Similarity score (lower is more similar for cosine distance).
        is_section: True if from DOCX (section), False if from PDF (page).
    """
    document_name: str
    page_number: int
    chunk_id: str
    text: str
    score: float
    is_section: bool = False
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Convert to context dictionary for answer generation.
        
        Returns:
            dict: Context format expected by the answer generator.
        """
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "is_section": self.is_section
        }


class Retriever:
    """
    Retrieves relevant document chunks for a query.
    
    Uses the vector store to find semantically similar chunks
    and formats them for the answer generator.
    """
    
    def __init__(self, top_k: int = None):
        """
        Initialize the retriever.
        
        Args:
            top_k: Number of chunks to retrieve. Defaults to settings value.
        """
        settings = get_settings()
        self.top_k = top_k or settings.top_k_results
        self.vector_store = VectorStore()
        
        logger.info(f"Retriever initialized with top_k={self.top_k}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        filter_document: str = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The user's question.
            top_k: Override for number of results to retrieve.
            filter_document: Optional document name to restrict search.
        
        Returns:
            List[RetrievedChunk]: Relevant chunks sorted by similarity.
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving top {k} chunks for query: {query[:50]}...")
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=k,
            filter_doc=filter_document
        )
        
        # Convert to RetrievedChunk objects
        retrieved = []
        for result in results:
            metadata = result.get("metadata", {})
            retrieved.append(RetrievedChunk(
                document_name=metadata.get("document_name", "unknown"),
                page_number=metadata.get("page_number", 0),
                chunk_id=result.get("chunk_id", "unknown"),
                text=result.get("text", ""),
                score=result.get("distance", 1.0),
                is_section=metadata.get("is_section", False)
            ))
        
        logger.info(f"Retrieved {len(retrieved)} chunks")
        return retrieved
    
    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as context string for the LLM.
        
        Args:
            chunks: List of retrieved chunks.
        
        Returns:
            str: Formatted context string with source markers.
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Chunk {i}]\n"
                f"Document: {chunk.document_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content:\n{chunk.text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def format_context_for_llm(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """
        Format retrieved chunks as structured data for the LLM prompt.
        
        Args:
            chunks: List of retrieved chunks.
        
        Returns:
            List[Dict]: List of chunk dictionaries with metadata.
        """
        return [chunk.to_context_dict() for chunk in chunks]
