"""
Unit tests for the TextChunker module.
"""

import pytest
from rag.chunker import TextChunker, TextChunk
from rag.document_loader import DocumentPage


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_small_text(self):
        """Test that small text creates single chunk."""
        pages = [DocumentPage(
            document_name="test.pdf",
            page_number=1,
            text="This is a short text."
        )]
        
        chunks = self.chunker.chunk_pages(pages)
        
        assert len(chunks) == 1
        assert chunks[0].document_name == "test.pdf"
        assert chunks[0].page_number == 1
        assert "short text" in chunks[0].text
    
    def test_chunk_large_text(self):
        """Test that large text creates multiple chunks."""
        long_text = "This is a sentence. " * 50  # Create long text
        pages = [DocumentPage(
            document_name="test.pdf",
            page_number=1,
            text=long_text
        )]
        
        chunks = self.chunker.chunk_pages(pages)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_name == "test.pdf"
            assert chunk.page_number == 1
            assert len(chunk.text) <= self.chunker.chunk_size + 50  # Allow some buffer
    
    def test_chunk_id_uniqueness(self):
        """Test that chunk IDs are unique."""
        text = "This is a sentence. " * 30
        pages = [DocumentPage(
            document_name="test.pdf",
            page_number=1,
            text=text
        )]
        
        chunks = self.chunker.chunk_pages(pages)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"
    
    def test_empty_pages(self):
        """Test that empty pages return no chunks."""
        chunks = self.chunker.chunk_pages([])
        assert len(chunks) == 0
    
    def test_to_metadata(self):
        """Test TextChunk.to_metadata method."""
        chunk = TextChunk(
            document_name="test.pdf",
            page_number=5,
            chunk_id="test_chunk_id",
            text="Sample text"
        )
        
        metadata = chunk.to_metadata()
        
        assert metadata["document_name"] == "test.pdf"
        assert metadata["page_number"] == 5
        assert metadata["chunk_id"] == "test_chunk_id"


class TestTextChunk:
    """Tests for TextChunk dataclass."""
    
    def test_text_chunk_creation(self):
        """Test creating a TextChunk object."""
        chunk = TextChunk(
            document_name="doc.pdf",
            page_number=3,
            chunk_id="doc_p3_c0_hash",
            text="Content here"
        )
        
        assert chunk.document_name == "doc.pdf"
        assert chunk.page_number == 3
        assert chunk.chunk_id == "doc_p3_c0_hash"
        assert chunk.text == "Content here"
