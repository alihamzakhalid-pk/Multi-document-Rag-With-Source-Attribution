"""
Unit tests for the DocumentLoader module.
"""

import pytest
from rag.document_loader import DocumentLoader, DocumentPage


class TestDocumentLoader:
    """Tests for DocumentLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
    
    def test_supported_extensions(self):
        """Test that supported file extensions are correctly identified."""
        assert DocumentLoader.is_supported("test.pdf")
        assert DocumentLoader.is_supported("test.PDF")
        assert DocumentLoader.is_supported("document.docx")
        assert DocumentLoader.is_supported("notes.txt")
        assert not DocumentLoader.is_supported("image.png")
        assert not DocumentLoader.is_supported("data.csv")
    
    def test_load_txt_from_content(self):
        """Test loading a TXT file from content bytes."""
        content = b"This is a test document.\nIt has multiple lines."
        pages = self.loader.load("test.txt", file_content=content)
        
        assert len(pages) == 1
        assert pages[0].document_name == "test.txt"
        assert pages[0].page_number == 1
        assert "test document" in pages[0].text
    
    def test_load_empty_content(self):
        """Test loading empty content returns no pages."""
        content = b"   "  # Only whitespace
        pages = self.loader.load("test.txt", file_content=content)
        
        assert len(pages) == 0
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            self.loader.load("test.xyz", file_content=b"content")
        
        assert "Unsupported file format" in str(excinfo.value)


class TestDocumentPage:
    """Tests for DocumentPage dataclass."""
    
    def test_document_page_creation(self):
        """Test creating a DocumentPage object."""
        page = DocumentPage(
            document_name="test.pdf",
            page_number=5,
            text="Sample content"
        )
        
        assert page.document_name == "test.pdf"
        assert page.page_number == 5
        assert page.text == "Sample content"
