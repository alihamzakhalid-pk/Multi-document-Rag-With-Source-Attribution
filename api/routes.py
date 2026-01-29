"""
API Routes for the Multi-Document RAG System.

Provides endpoints for document management and question answering.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from utils.logger import get_logger
from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.answer_generator import AnswerGenerator

from .schemas import (
    QueryRequest,
    QueryResponse,
    SourceReference,
    RetrievedChunkInfo,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    HealthResponse,
    ErrorResponse,
)

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Initialize components (lazy loading)
_document_loader = None
_chunker = None
_vector_store = None
_retriever = None
_answer_generator = None


def get_document_loader() -> DocumentLoader:
    """Get or create DocumentLoader instance."""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader


def get_chunker() -> TextChunker:
    """Get or create TextChunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker()
    return _chunker


def get_vector_store() -> VectorStore:
    """Get or create VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_retriever() -> Retriever:
    """Get or create Retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_answer_generator() -> AnswerGenerator:
    """Get or create AnswerGenerator instance."""
    global _answer_generator
    if _answer_generator is None:
        _answer_generator = AnswerGenerator()
    return _answer_generator


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the RAG system."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the system including
    document and chunk counts.
    """
    try:
        vector_store = get_vector_store()
        docs = vector_store.get_all_documents()
        total_chunks = vector_store.count()
        
        return HealthResponse(
            status="healthy",
            documents_loaded=len(docs),
            total_chunks=total_chunks
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            documents_loaded=0,
            total_chunks=0
        )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload Document",
    description="Upload a document (PDF, DOCX, or TXT) to be indexed for RAG queries."
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload")
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    The document will be parsed, chunked, and stored in the vector store
    for retrieval during queries.
    """
    # Validate file type
    if not DocumentLoader.is_supported(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported formats: PDF, DOCX, TXT"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Load document
        loader = get_document_loader()
        pages = loader.load(file.filename, file_content=content)
        
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content could be extracted from the document"
            )
        
        # Chunk pages
        chunker = get_chunker()
        chunks = chunker.chunk_pages(pages)
        
        # Add to vector store
        vector_store = get_vector_store()
        chunks_added = vector_store.add_chunks(chunks)
        
        logger.info(
            f"Uploaded document: {file.filename}, "
            f"pages: {len(pages)}, chunks: {chunks_added}"
        )
        
        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            document_name=file.filename,
            pages_processed=len(pages),
            chunks_created=chunks_added
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Get a list of all uploaded documents."
)
async def list_documents() -> DocumentListResponse:
    """
    List all documents in the system.
    
    Returns document names and chunk counts.
    """
    try:
        vector_store = get_vector_store()
        doc_names = vector_store.get_all_documents()
        
        documents = []
        for name in doc_names:
            chunks = vector_store.get_document_chunks(name)
            documents.append(DocumentInfo(
                document_name=name,
                chunk_count=len(chunks)
            ))
        
        return DocumentListResponse(
            documents=documents,
            total_chunks=vector_store.count()
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.delete(
    "/documents/{document_name}",
    response_model=DocumentDeleteResponse,
    summary="Delete Document",
    description="Delete a document and all its chunks from the system."
)
async def delete_document(document_name: str) -> DocumentDeleteResponse:
    """
    Delete a document from the system.
    
    All chunks belonging to the document will be removed
    from the vector store.
    """
    try:
        vector_store = get_vector_store()
        
        # Check if document exists
        existing_docs = vector_store.get_all_documents()
        if document_name not in existing_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_name}' not found"
            )
        
        # Delete document
        chunks_deleted = vector_store.delete_document(document_name)
        
        logger.info(f"Deleted document: {document_name}, chunks: {chunks_deleted}")
        
        return DocumentDeleteResponse(
            message="Document deleted successfully",
            document_name=document_name,
            chunks_deleted=chunks_deleted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Documents",
    description="Ask a question to be answered using the uploaded documents."
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the document store.
    
    Retrieves relevant chunks and generates an answer based
    ONLY on the retrieved context. Returns the answer with
    source attributions.
    """
    try:
        # Initialize components
        retriever = get_retriever()
        answer_generator = get_answer_generator()
        
        # Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            filter_document=request.filter_document
        )
        
        # Generate answer
        answer_response = answer_generator.generate(
            question=request.question,
            retrieved_chunks=retrieved_chunks
        )
        
        # Convert to API response
        sources = [
            SourceReference(
                document_name=s.document_name,
                page=s.page,
                chunk_id=s.chunk_id,
                is_section=getattr(s, 'is_section', False)
            )
            for s in answer_response.sources
        ]
        
        # Convert retrieved chunks to API format
        retrieved_chunks_info = [
            RetrievedChunkInfo(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score,
                is_section=getattr(chunk, 'is_section', False)
            )
            for chunk in retrieved_chunks
        ]
        
        return QueryResponse(
            answer=answer_response.answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks_info
        )
        
    except RuntimeError as e:
        # API key not configured
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )
