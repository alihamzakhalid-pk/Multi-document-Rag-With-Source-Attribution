"""
Document Converter for the Multi-Document RAG System.

Converts DOCX files to PDF using docx2pdf to enable accurate page number extraction.
Note: docx2pdf requires Microsoft Word to be installed on Windows.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentConverter:
    """
    Converts documents to PDF format for accurate page extraction.
    
    Uses docx2pdf which requires Microsoft Word on Windows.
    """
    
    def __init__(self):
        """Initialize the document converter."""
        self._available = None
        logger.info("DocumentConverter initialized")
    
    def is_available(self) -> bool:
        """
        Check if docx2pdf conversion is available.
        
        Returns:
            bool: True if docx2pdf is available and working.
        """
        if self._available is not None:
            return self._available
        
        try:
            import docx2pdf
            self._available = True
            logger.info("docx2pdf is available for document conversion")
        except ImportError:
            self._available = False
            logger.warning("docx2pdf not installed. DOCX files will use section-based parsing.")
        
        return self._available
    
    def convert_docx_to_pdf(
        self, 
        file_content: bytes, 
        original_filename: str
    ) -> Optional[bytes]:
        """
        Convert a DOCX file to PDF.
        
        Args:
            file_content: The DOCX file content as bytes.
            original_filename: Original filename (for logging).
            
        Returns:
            bytes: The converted PDF content, or None if conversion fails.
        """
        if not self.is_available():
            logger.warning(f"Cannot convert {original_filename}: docx2pdf not available")
            return None
        
        try:
            from docx2pdf import convert
            
            # Create temporary files for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_docx = Path(temp_dir) / "input.docx"
                temp_pdf = Path(temp_dir) / "input.pdf"
                
                # Write DOCX content to temp file
                temp_docx.write_bytes(file_content)
                
                # Convert to PDF
                logger.info(f"Converting {original_filename} to PDF...")
                convert(str(temp_docx), str(temp_pdf))
                
                # Read PDF content
                if temp_pdf.exists():
                    pdf_content = temp_pdf.read_bytes()
                    logger.info(f"Successfully converted {original_filename} to PDF ({len(pdf_content)} bytes)")
                    return pdf_content
                else:
                    logger.error(f"PDF output file not created for {original_filename}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error converting {original_filename} to PDF: {str(e)}")
            return None
    
    def needs_conversion(self, file_path: str) -> bool:
        """
        Check if a file needs to be converted to PDF.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            bool: True if the file is a DOCX that should be converted.
        """
        extension = Path(file_path).suffix.lower()
        return extension == ".docx"
