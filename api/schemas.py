"""
Pydantic schemas for the Multi-Document RAG API.

Defines request and response models for API validation
and documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """
    Reference to a source document chunk.
    
    Used in query responses to attribute information
    to specific document locations.
    """
    document_name: str = Field(
        ...,
        description="Name of the source document"
    )
    page: int = Field(
        ...,
        description="Page number (PDF) or section number (DOCX)"
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the chunk"
    )
    is_section: bool = Field(
        default=False,
        description="True if this is a section (DOCX), False if page (PDF)"
    )


class QueryRequest(BaseModel):
    """
    Request model for document queries.
    """
    question: str = Field(
        ...,
        description="The question to answer based on uploaded documents",
        min_length=1,
        max_length=2000
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve (overrides default)",
        ge=1,
        le=20
    )
    filter_document: Optional[str] = Field(
        default=None,
        description="Optional: restrict search to a specific document"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the main findings of the study?",
                    "top_k": 5,
                    "filter_document": None
                }
            ]
        }
    }


class RetrievedChunkInfo(BaseModel):
    """
    Information about a retrieved chunk (for debug/transparency).
    """
    document_name: str = Field(
        ...,
        description="Name of the source document"
    )
    page_number: int = Field(
        ...,
        description="Page number (PDF) or section number (DOCX)"
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the chunk"
    )
    text: str = Field(
        ...,
        description="The chunk text content"
    )
    score: float = Field(
        ...,
        description="Similarity score (lower is more similar)"
    )
    is_section: bool = Field(
        default=False,
        description="True if this is a section (DOCX), False if page (PDF)"
    )


class QueryResponse(BaseModel):
    """
    Response model for document queries.
    
    Contains the generated answer and source attributions.
    """
    answer: str = Field(
        ...,
        description="The generated answer based on retrieved documents"
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="List of source references for the answer"
    )
    retrieved_chunks: List[RetrievedChunkInfo] = Field(
        default_factory=list,
        description="Retrieved chunks used for context (for debugging/transparency)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The study found that...",
                    "sources": [
                        {
                            "document_name": "research_paper.pdf",
                            "page": 5,
                            "chunk_id": "research_paper.pdf_p5_c0_abc123"
                        }
                    ],
                    "retrieved_chunks": []
                }
            ]
        }
    }


class DocumentInfo(BaseModel):
    """
    Information about an uploaded document.
    """
    document_name: str = Field(
        ...,
        description="Name of the document"
    )
    chunk_count: int = Field(
        ...,
        description="Number of chunks created from the document"
    )


class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload.
    """
    message: str = Field(
        ...,
        description="Status message"
    )
    document_name: str = Field(
        ...,
        description="Name of the uploaded document"
    )
    pages_processed: int = Field(
        ...,
        description="Number of pages/sections processed"
    )
    chunks_created: int = Field(
        ...,
        description="Number of text chunks created"
    )


class DocumentListResponse(BaseModel):
    """
    Response model for listing documents.
    """
    documents: List[DocumentInfo] = Field(
        ...,
        description="List of uploaded documents with their info"
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks across all documents"
    )


class DocumentDeleteResponse(BaseModel):
    """
    Response model for document deletion.
    """
    message: str = Field(
        ...,
        description="Status message"
    )
    document_name: str = Field(
        ...,
        description="Name of the deleted document"
    )
    chunks_deleted: int = Field(
        ...,
        description="Number of chunks removed"
    )


class HealthResponse(BaseModel):
    """
    Response model for health check.
    """
    status: str = Field(
        ...,
        description="Service status"
    )
    documents_loaded: int = Field(
        ...,
        description="Number of documents in the store"
    )
    total_chunks: int = Field(
        ...,
        description="Total chunks in the store"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details"
    )
