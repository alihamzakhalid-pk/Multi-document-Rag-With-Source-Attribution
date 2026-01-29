"""
Text Chunker for the Multi-Document RAG System.

Handles splitting documents into smaller chunks suitable for embedding
and retrieval, while preserving source metadata for attribution.
"""

from dataclasses import dataclass
from typing import List
import hashlib

from config.settings import get_settings
from utils.logger import get_logger
from .document_loader import DocumentPage

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """
    Represents a chunk of text with full source attribution.
    
    Attributes:
        document_name: Name of the source document.
        page_number: Page number from the original document.
        chunk_id: Unique identifier for this chunk.
        text: The chunked text content.
        is_section: True if from DOCX (section), False if from PDF (page).
    """
    document_name: str
    page_number: int
    chunk_id: str
    text: str
    is_section: bool = False
    
    def to_metadata(self) -> dict:
        """
        Convert chunk attributes to a metadata dictionary.
        
        Returns:
            dict: Metadata suitable for vector store storage.
        """
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "is_section": self.is_section,
        }


class TextChunker:
    """
    Splits document pages into smaller, overlapping text chunks.
    
    Uses configurable chunk size and overlap to create chunks that
    maintain context while being suitable for embedding models.
    """
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk. Defaults to settings value.
            chunk_overlap: Number of overlapping characters. Defaults to settings value.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        logger.info(
            f"TextChunker initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def chunk_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """
        Split multiple document pages into chunks.
        
        Args:
            pages: List of DocumentPage objects to chunk.
        
        Returns:
            List[TextChunk]: All chunks from all pages.
        """
        all_chunks = []
        
        for page in pages:
            chunks = self._chunk_text(
                text=page.text,
                document_name=page.document_name,
                page_number=page.page_number,
                is_section=getattr(page, 'is_section', False)
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def _chunk_text(
        self, 
        text: str, 
        document_name: str, 
        page_number: int,
        is_section: bool = False
    ) -> List[TextChunk]:
        """
        Split a single text into chunks with overlap.
        
        Args:
            text: The text content to split.
            document_name: Source document name.
            page_number: Source page number.
            is_section: True if this is a section (DOCX), False if page (PDF).
        
        Returns:
            List[TextChunk]: Chunks from this text.
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            chunk_id = self._generate_chunk_id(document_name, page_number, 0)
            chunks.append(TextChunk(
                document_name=document_name,
                page_number=page_number,
                chunk_id=chunk_id,
                text=text,
                is_section=is_section
            ))
            return chunks
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                boundary = self._find_boundary(text, end)
                if boundary > start:
                    end = boundary
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = self._generate_chunk_id(document_name, page_number, chunk_index)
                chunks.append(TextChunk(
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    is_section=is_section
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text) - 1:
                break
        
        return chunks
    
    def _find_boundary(self, text: str, position: int) -> int:
        """
        Find a natural text boundary (sentence or word) near the position.
        
        Args:
            text: The full text.
            position: Target position to break near.
        
        Returns:
            int: Adjusted position at a natural boundary.
        """
        # Search window
        search_start = max(0, position - 50)
        search_end = min(len(text), position + 50)
        search_text = text[search_start:search_end]
        
        # Try to find sentence boundaries first
        sentence_ends = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        best_break = None
        
        for end_marker in sentence_ends:
            idx = search_text.rfind(end_marker, 0, position - search_start + 50)
            if idx != -1:
                actual_pos = search_start + idx + len(end_marker)
                if best_break is None or actual_pos > best_break:
                    best_break = actual_pos
        
        if best_break:
            return best_break
        
        # Fall back to word boundary
        word_end = position
        while word_end < len(text) and text[word_end] not in " \n\t":
            word_end += 1
        
        return word_end
    
    def _generate_chunk_id(
        self, 
        document_name: str, 
        page_number: int, 
        chunk_index: int
    ) -> str:
        """
        Generate a unique chunk ID.
        
        Args:
            document_name: Source document name.
            page_number: Source page number.
            chunk_index: Index of chunk within the page.
        
        Returns:
            str: Unique chunk identifier.
        """
        # Create a deterministic but unique ID
        base = f"{document_name}_page{page_number}_chunk{chunk_index}"
        # Add a short hash for extra uniqueness
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"{document_name}_p{page_number}_c{chunk_index}_{hash_suffix}"
